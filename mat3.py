from __future__ import annotations
from dataclasses import dataclass
import math as m
# from cmu_graphics import *  # cmu general
from cmu_graphics import cmu_graphics as c  # c.run()


CMU_RUN = True

f = 250
screen_center = (200, 200)  # screen coord


### DEFINE vector(1x3), matrix(3x3)
@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

@dataclass(frozen=True)
class Mat3:
    m: tuple[tuple[float, float, float],
             tuple[float, float, float],
             tuple[float, float, float]]

    def __matmul__(self, other):
        # Mat3 @ Vec3  (apply transform)
        if isinstance(other, Vec3):
            x, y, z = other.x, other.y, other.z
            return Vec3(
                self.m[0][0]*x + self.m[0][1]*y + self.m[0][2]*z,
                self.m[1][0]*x + self.m[1][1]*y + self.m[1][2]*z,
                self.m[2][0]*x + self.m[2][1]*y + self.m[2][2]*z,
            )

        # Mat3 @ Mat3 (compose transforms)
        if isinstance(other, Mat3):
            A, B = self.m, other.m

            def cell(i, j):
                # (A B)_{ij} = sum_k A_{ik} B_{kj}
                return A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]

            return Mat3((
                (cell(0, 0), cell(0, 1), cell(0, 2)),
                (cell(1, 0), cell(1, 1), cell(1, 2)),
                (cell(2, 0), cell(2, 1), cell(2, 2)),
            ))

        return NotImplemented

### CAMERA
cam_cord = Vec3(0, 0, -200)
cam_rotation = Vec3(0, 0, 0)

class Camera:
    def __init__(self, cord: Vec3, rotation: Vec3):
        self.cord = cord
        self.rotation = rotation
        self._initial_cord = cord
        self._initial_rotation = rotation

    def _sync_globals(self):
        globals()['cam_cord'] = self.cord
        globals()['cam_rotation'] = self.rotation

    def _sync_objects(self):
        for entry in _objects.values():
            obj = entry.get('obj') if isinstance(entry, dict) else entry
            if obj is None:
                continue
            if hasattr(obj, 'cam_cord'):
                obj.cam_cord = self.cord
            if hasattr(obj, 'cam_rotation'):
                obj.cam_rotation = self.rotation
            if hasattr(obj, 'redraw'):
                obj.redraw()
        drawAxis()
        depth_layer.update(_objects, self.cord, self.rotation, _axis_group)

    def move(self, delta: Vec3):
        self.cord = self.cord + delta
        self._sync_globals()
        self._sync_objects()

    def rotate(self, delta: Vec3):
        self.rotation = Vec3(
            self.rotation.x + delta.x,
            self.rotation.y + delta.y,
            self.rotation.z + delta.z,
        )
        rx = 0 if abs(self.rotation.x) <= 1e-6 else self.rotation.x
        ry = 0 if abs(self.rotation.y) <= 1e-6 else self.rotation.y
        rz = 0 if abs(self.rotation.z) <= 1e-6 else self.rotation.z
        rx = rx % (2 * m.pi)
        ry = ry % (2 * m.pi)
        rz = rz % (2 * m.pi)
        self.rotation = Vec3(rx, ry, rz)
        self._sync_globals()
        self._sync_objects()

    def reset(self):
        self.cord = self._initial_cord
        self.rotation = self._initial_rotation
        self._sync_globals()
        self._sync_objects()

camera = Camera(cam_cord, cam_rotation)


### ROTATION
# theta must be RAD
def Rx(theta: float) -> Mat3:
    c, s = m.cos(theta), m.sin(theta)
    return Mat3((
        (1.0, 0.0, 0.0),
        (0.0,  c, -s),
        (0.0,  s,  c),
    ))
def Ry(theta: float) -> Mat3:
    c, s = m.cos(theta), m.sin(theta)
    return Mat3((
        ( c, 0.0,  s),
        (0.0, 1.0, 0.0),
        (-s, 0.0,  c),
    ))
def Rz(theta: float) -> Mat3:
    c, s = m.cos(theta), m.sin(theta)
    return Mat3((
        ( c, -s, 0.0),
        ( s,  c, 0.0),
        (0.0, 0.0, 1.0),
    ))

def rotateVerts(verts: list[Vec3], rotation: Vec3) -> list[Vec3]:
    rx, ry, rz = rotation.x, rotation.y, rotation.z
    R = Rz(rz) @ Ry(ry) @ Rx(rx)    # Order matters
    return [R @ v for v in verts]


### PROJECTION
# 내 목표는: x, y, z 기반의 카메라 고정 좌표계에서 p, q, r 기반의 카메라 좌표계로 변환, 이후 u, v 기반의 화면 좌표계로 변환
# 따라서, 기존의 cam 변수는: tuple[float, float]에서 Vec3으로 바꿔야됨.
# 또, 기존의 cam 변수는: 화면 좌표계 기준에서의 중앙을 의미했으나, 이를 카메라 좌표계의 좌표로 바꿔야됨.
# 따라서, 기존의 cam 변수에서 중앙을 의미하는 부분과 실질적인 카메라 좌표를 의미하는 부분을 분기할 필요성 제기.
# 기존의 중앙을 의미하는 부분은 screen_center 변수로 따로 정의.
# 이후, projectToCamera 함수를 새로 정의하여 카메라 좌표계로 변환, 이후 projectToScreen 함수로 화면 좌표계로 변환.
# cam_cord, cam_rotation 변수를 새로 정의.

def projectToCamera(
    points: Vec3,
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None
) -> Vec3:
    """
    WORLD COORD SYS: x, y, z  ->  CAMERA COORD SYS: p, q, r
    """
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    local = points - cam_cord
    if cam_rotation == Vec3(0, 0, 0):
        return local
    rx, ry, rz = cam_rotation.x, cam_rotation.y, cam_rotation.z
    # Inverse rotation to move world space into camera space
    R = Rx(-rx) @ Ry(-ry) @ Rz(-rz)
    return R @ local
def projectToScreen(
    points: Vec3,
    screen_center: tuple[float, float] | None = None
) -> tuple[float, float] | None:
    """
    CAMERA COORD SYS: p, q, r  ->  SCREEN COORD SYS: u, v

    MUST: z > 0
    """
    # camera looks along +z and points.z must be > 0
    # u = cx + f * (x / z)
    # v = cy - f * (y / z)
    if points.z <= 0:
        return None  # Behind camera or at camera plane
    if screen_center is None:
        screen_center = globals()['screen_center']
    cx, cy = screen_center
    u = cx + f * (points.x / points.z)
    v = cy - f * (points.y / points.z)
    return (u, v)

def _clip_segment_to_near_plane(
    p0: Vec3, p1: Vec3, near: float = 1e-6
) -> tuple[Vec3, Vec3] | None:
    """
    Clip a line segment (in camera space) to the near plane.
    Returns (clipped_p0, clipped_p1) if any part is visible, else None.
    """
    z0, z1 = p0.z, p1.z
    if z0 > near and z1 > near:
        return (p0, p1)
    if z0 <= near and z1 <= near:
        return None
    # One visible, one behind: P(t) = p0 + t*(p1-p0) where z = near
    t = (near - z0) / (z1 - z0)
    clipped = Vec3(
        p0.x + t * (p1.x - p0.x),
        p0.y + t * (p1.y - p0.y),
        near,
    )
    if z0 > near:
        return (p0, clipped)
    return (clipped, p1)


### MANAGEMENT
_objects: dict[str, dict[str, object]] = {}
_selected_index = 0
_axis_group: c.Group | None = None
_obj_order: list[int, object] = []

def register_object(obj: object, name: str, obj_type: str | None = None):
    _obj_order.append((len(_obj_order), obj))
    _objects[name] = {
        'obj': obj,
        'type': obj_type if obj_type is not None else type(obj).__name__,
    }

def get_selected_object(kind: str):
    if not _objects:
        return None
    if kind == 'obj':
        return list(_objects.values())[_selected_index]['obj']
    elif kind == 'name':
        return list(_objects.keys())[_selected_index]
    elif kind == 'type':
        return list(_objects.values())[_selected_index].get('type')
    else:
        return None

def select_next_object():
    global _selected_index
    if _objects:
        _selected_index = (_selected_index + 1) % len(_objects)

def clearObject(name: str):
    global _selected_index
    entry = _objects.pop(name, None)
    if entry is not None:
        obj = entry['obj'] if isinstance(entry, dict) and 'obj' in entry else entry
        if isinstance(obj, c.Group):
            obj.clear()
        else:
            group = getattr(obj, 'group', None)
            if group is not None and hasattr(group, 'clear'):
                group.clear()
            elif hasattr(obj, 'visible'):
                obj.visible = False
    if _selected_index >= len(_objects):
        _selected_index = 0


### OBJECTS
# Axis
SHOW_AXIS = True
def drawAxis():
    global _axis_group
    if not SHOW_AXIS:
        if _axis_group is not None:
            _axis_group.clear()
            _axis_group.visible = False
        return None

    # ALL axes intersect at (0,0,0)
    axis_len = 200
    axis_pairs = [
        (Vec3(-axis_len, 0, 0), Vec3(+axis_len, 0, 0), 'red'),    # x
        (Vec3(0, -axis_len, 0), Vec3(0, +axis_len, 0), 'green'),  # y
        (Vec3(0, 0, -axis_len + 1), Vec3(0, 0, +axis_len), 'blue'),  # z
    ]

    if _axis_group is None:
        _axis_group = c.Group()
    _axis_group.clear()
    _axis_group.visible = True

    for p0_world, p1_world, color in axis_pairs:
        p0_cam = projectToCamera(p0_world); p1_cam = projectToCamera(p1_world)
        clipped = _clip_segment_to_near_plane(p0_cam, p1_cam)
        if clipped is None:
            continue
        q0 = projectToScreen(clipped[0]); q1 = projectToScreen(clipped[1])
        if q0 is None or q1 is None:
            continue
        _axis_group.add(
            c.Line(q0[0], q0[1], q1[0], q1[1],
                   fill=color, dashes=True, arrowEnd=True, opacity=50)
        )
    return _axis_group

# Cuboid
def _cuboid_faces(
    cord: Vec3,
    size: Vec3,
    rotation: Vec3 | None = None,
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None,
    screen_center: tuple[float, float] | None = None,
    fill: list[str, str, str, str, str, str] | None = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightyellow']
) -> list[c.Polygon] | None:
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    if screen_center is None:
        screen_center = globals()['screen_center']

    w = size.x if size.x > 0 else 1e-6
    h = size.y if size.y > 0 else 1e-6
    d = size.z if size.z > 0 else 1e-6
    w = w / 2.0; h = h / 2.0; d = d / 2.0; 
    verts = [
        Vec3(-w, -h, -d), Vec3(+w, -h, -d),
        Vec3(+w, +h, -d), Vec3(-w, +h, -d),
        Vec3(-w, -h, +d), Vec3(+w, -h, +d),
        Vec3(+w, +h, +d), Vec3(-w, +h, +d)
    ]
    if rotation is not None and rotation != Vec3(0, 0, 0):
        verts = rotateVerts(verts, rotation)
    verts = [v + cord for v in verts]
    
    camera_verts = [projectToCamera(v, cam_cord, cam_rotation) for v in verts]
    points = [projectToScreen(v, screen_center) for v in camera_verts]
    
    # Ignore vert beyond cam
    if any(coord is None for coord in points):
        return None
    
    center_cam = projectToCamera(cord, cam_cord, cam_rotation)
    center_point = projectToScreen(center_cam, screen_center)
    if center_point is None:
        return None
    
    try:
        # Front face (vertices 4, 5, 6, 7)
        front_face = c.Polygon(points[4][0], points[4][1],
                points[5][0], points[5][1],
                points[6][0], points[6][1],
                points[7][0], points[7][1],
                fill=fill[0], border='black', borderWidth=1, opacity=50)
        # Back face (vertices 0, 1, 2, 3)
        back_face = c.Polygon(points[0][0], points[0][1],
                points[1][0], points[1][1],
                points[2][0], points[2][1],
                points[3][0], points[3][1],
                fill=fill[1], border='black', borderWidth=1, opacity=50)
        # Top face (vertices 3, 2, 6, 7)
        top_face = c.Polygon(points[3][0], points[3][1],
                points[2][0], points[2][1],
                points[6][0], points[6][1],
                points[7][0], points[7][1],
                fill=fill[2], border='black', borderWidth=1, opacity=50)
        # Bottom face (vertices 0, 1, 5, 4)
        bottom_face = c.Polygon(points[0][0], points[0][1],
                points[1][0], points[1][1],
                points[5][0], points[5][1],
                points[4][0], points[4][1],
                fill=fill[3], border='black', borderWidth=1, opacity=50)
        # Right face (vertices 1, 2, 6, 5)
        right_face = c.Polygon(points[1][0], points[1][1],
                points[2][0], points[2][1],
                points[6][0], points[6][1],
                points[5][0], points[5][1],
                fill=fill[4], border='black', borderWidth=1, opacity=50)
        # Left face (vertices 0, 3, 7, 4)
        left_face = c.Polygon(points[0][0], points[0][1],
                points[3][0], points[3][1],
                points[7][0], points[7][1],
                points[4][0], points[4][1],
                fill=fill[5], border='black', borderWidth=1, opacity=50)
    
        return [
            front_face,
            back_face,
            top_face,
            bottom_face,
            right_face,
            left_face,
        ]
    except: 
        return None

class Cuboid:
    """
    WORLD (x, y, z)

    - redraw
    - TRANSLATION
        - move (delta: Vec3)
        - set_cord (cord: Vec3)
        - reset_cord
    - SCALING
        - scale (delta: Vec3)
        - set_size (size: Vec3)
        - reset_size
    - ROTATION
        - rotate (delta: Vec3)
        - set_rotation (rotation: Vec3)
        - reset_rotation
    """
    def __init__(
        self,
        cord: Vec3,
        size: Vec3,
        rotation: Vec3 | None = None,
        cam_cord: Vec3 | None = None,
        cam_rotation: Vec3 | None = None,
        screen_center: tuple[float, float] | None = None,
    ):
        if cam_cord is None:
            cam_cord = globals()['cam_cord']
        if cam_rotation is None:
            cam_rotation = globals()['cam_rotation']
        if screen_center is None:
            screen_center = globals()['screen_center']
        self.cord = cord
        self._initial_cord = cord
        self.size = size
        self._initial_size = size
        self.rotation = rotation if rotation is not None else Vec3(0, 0, 0)
        self._initial_rotation = self.rotation
        self.cam_cord = cam_cord
        self.cam_rotation = cam_rotation
        self.screen_center = screen_center
        self.group = c.Group()
        self.redraw()

    def redraw(self):
        faces = _cuboid_faces(
            self.cord,
            self.size,
            self.rotation,
            self.cam_cord,
            self.cam_rotation,
            self.screen_center
        )
        self.group.clear()
        if faces is None:
            self.group.visible = False
            return self.group
        self.group.visible = True
        for face in faces:
            self.group.add(face)
        return self.group

    # Translation
    def move(self, delta: Vec3):
        self.cord = self.cord + delta
        return self.redraw()

    def set_cord(self, cord: Vec3):
        self.cord = cord
        return self.redraw()
    def reset_cord(self):
        self.cord = self._initial_cord
        return self.redraw()
    
    # Scale
    def scale(self, delta: Vec3):
        self.size = Vec3(
            self.size.x + delta.x,
            self.size.y + delta.y,
            self.size.z + delta.z,
        )
        return self.redraw()

    def set_size(self, size: Vec3):
        self.size = size
        return self.redraw()
    def reset_size(self):
        self.size = self._initial_size
        return self.redraw()

    # Rotation (rad)
    def rotate(self, delta: Vec3):
        self.rotation = Vec3(
            self.rotation.x + delta.x,
            self.rotation.y + delta.y,
            self.rotation.z + delta.z,
        )
        rx = 0 if abs(self.rotation.x) <= 1e-6 else self.rotation.x
        ry = 0 if abs(self.rotation.y) <= 1e-6 else self.rotation.y
        rz = 0 if abs(self.rotation.z) <= 1e-6 else self.rotation.z
        rx = rx % (2 * m.pi)
        ry = ry % (2 * m.pi)
        rz = rz % (2 * m.pi)
        self.rotation = Vec3(rx, ry, rz)
        return self.redraw()

    def set_rotation(self, rotation: Vec3):
        self.rotation = rotation
        return self.redraw()
    def reset_rotation(self):
        self.rotation = self._initial_rotation
        return self.redraw()

# Ellipsoid
    # c.Arc(x, y, w, h, start, end, fill=None, border=None, borderWidth=1, opacity=100)
# Sphere


class DepthLayerManager:
    """Manages render order by camera-space depth using toFront/toBack."""

    def _depth_z(self, obj, cam_cord: Vec3, cam_rotation: Vec3) -> float:
        ref = getattr(obj, 'cord', None)
        if ref is None:
            return float('inf')
        p = projectToCamera(ref, cam_cord, cam_rotation)
        return p.z if p.z > 0 else float('inf')

    def update(
        self,
        obj_registry: dict,
        cam_cord: Vec3,
        cam_rotation: Vec3,
        axis_group: c.Group | None = None,
    ) -> None:
        entries = [(n, e['obj']) for n, e in obj_registry.items()]
        # drawable: obj.group if exists, else obj itself (shape/Group auto in app.group per CMU docs)
        drawable_get = lambda o: getattr(o, 'group', o)
        entries = [
            (n, o)
            for n, o in entries
            if (d := drawable_get(o)) is not None
            and hasattr(d, 'toFront')
            and getattr(d, 'visible', True)
        ]
        sorted_entries = sorted(
            entries,
            key = lambda e: self._depth_z(e[1], cam_cord, cam_rotation),
            reverse = True,
        )
        for _, obj in sorted_entries:
            drawable_get(obj).toFront()
        if axis_group is not None and getattr(axis_group, 'visible', False):
            axis_group.toBack()


### UI
def ask(prompt: str) -> str:
    return c.app.getTextInput(prompt)
def alert(message: str | None):
    if message is None:
        return
    c.app.showMessage(message)


class InputInfo:
    """
    UI for input

    - Keyboard: keys | modifiers
    - Mouse: x, y | button

    Pos: (u, v) = (2, 398), left-bottom
    """
    def __init__(
        self,
        key_x: float = 2,
        key_y: float = 398,
        mouse_x: float = 2,
        mouse_y: float = 388,
    ):
        self.key_x = key_x
        self.key_y = key_y
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
        self.label_keyboard = c.Label(
            self._format_keyboard(None, None),
            key_x, key_y,
            fill='black', align='left-bottom', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_mouse = c.Label(
            self._format_mouse(None, None, None),
            mouse_x, mouse_y,
            fill='black', align='left-bottom', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.group = c.Group(self.label_keyboard, self.label_mouse)

    def _format_value(self, value):
        if value is None:
            return 'None'
        if value == '':
            return 'None'
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            return 'None'
        return str(value)

    def _format_keyboard(self, keys, modifiers):
        return f'{self._format_value(keys)} | {self._format_value(modifiers)}'

    def _format_mouse(self, x, y, button=None):
        return f'{self._format_value(x)}, {self._format_value(y)} | {self._format_value(button)}'

    def set_keyboard(self, keys=None, modifiers=None):
        self.label_keyboard.value = self._format_keyboard(keys, modifiers)
        self.label_keyboard.left = self.key_x
        self.label_keyboard.bottom = self.key_y

    def set_mouse(self, x: float | None = None, y: float | None = None, button: int | None = None):
        self.label_mouse.value = self._format_mouse(x, y, button)
        self.label_mouse.left = self.mouse_x
        self.label_mouse.bottom = self.mouse_y

    def set_mouse_pressed(self, pressed: bool):
        self.label_mouse.bold = not bool(pressed)

class CameraInfo:
    """
    UI for camera

    - Position (x, y, z)
    - Rotation (rx, ry, rz)

    Pos: (u, v) = (398, 388), right-bottom
    """
    def __init__(self, x: float = 398, y: float = 388):
        self.base_x = x
        self.base_y = y
        self.label_camera_position_x = c.Label(
            'X',
            x, y,
            fill='red', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_camera_position_y = c.Label(
            'Y',
            x, y,
            fill='green', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_camera_position_z = c.Label(
            'Z',
            x, y,
            fill='blue', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_camera_rotation_rx = c.Label(
            'RX',
            x, y,
            fill='red', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_camera_rotation_ry = c.Label(
            'RY',
            x, y,
            fill='green', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_camera_rotation_rz = c.Label(
            'RZ',
            x, y,
            fill='blue', align='right', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.group = c.Group(
            self.label_camera_position_x,
            self.label_camera_position_y,
            self.label_camera_position_z,
            self.label_camera_rotation_rx,
            self.label_camera_rotation_ry,
            self.label_camera_rotation_rz,
        )
        self.update()

    def _anchor_all_labels(self):
        # position
        self.label_camera_position_x.right = self.base_x - 5 # x
        self.label_camera_position_x.top = self.base_y - 60
        self.label_camera_position_y.right = self.base_x - 5 # y
        self.label_camera_position_y.top = self.base_y - 50
        self.label_camera_position_z.right = self.base_x - 5 # z
        self.label_camera_position_z.top = self.base_y - 40
        # rotation
        self.label_camera_rotation_rx.right = self.base_x - 5 # rx
        self.label_camera_rotation_rx.top = self.base_y - 20
        self.label_camera_rotation_ry.right = self.base_x - 5 # ry
        self.label_camera_rotation_ry.top = self.base_y - 10
        self.label_camera_rotation_rz.right = self.base_x - 5 # rz
        self.label_camera_rotation_rz.top = self.base_y
    def update(self):
        self.label_camera_position_x.value = str(camera.cord.x)
        self.label_camera_position_y.value = str(camera.cord.y)
        self.label_camera_position_z.value = str(camera.cord.z)
        self.label_camera_rotation_rx.value = str(camera.rotation.x) + ' (' + str(round(camera.rotation.x * 180.0 / m.pi)) + 'º)'
        self.label_camera_rotation_ry.value = str(camera.rotation.y) + ' (' + str(round(camera.rotation.y * 180.0 / m.pi)) + 'º)'
        self.label_camera_rotation_rz.value = str(camera.rotation.z) + ' (' + str(round(camera.rotation.z * 180.0 / m.pi)) + 'º)'
        self._anchor_all_labels()

class SelectedObjectInfo:
    """
    UI for selected object

    - Selected name
    - Object type
    - Position (x, y, z)
    - Size (w, h, d)
    - Rotation (rx, ry, rz)

    Pos: (u, v) = (2, 2), left-top
    """
    def __init__(self, x: float = 2, y: float = 2):
        self.base_x = x
        self.base_y = y
        self.label_selected = c.Label(
            'SELECTED NAME',
            x, y,
            fill='black', align='left-top', bold=True,
            size=15,
            opacity=80,
        )
        self.label_type = c.Label(
            '-',
            x, y,
            fill='black', align='left-bottom', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_x = c.Label(
            'X',
            x, y,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_y = c.Label(
            'Y',
            x, y,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_z = c.Label(
            'Z',
            x, y,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_w = c.Label(
            'W',
            x, y,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_h = c.Label(
            'H',
            x, y,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_d = c.Label(
            'D',
            x, y,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_rx = c.Label(
            'RX | deg',
            x, y,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_ry = c.Label(
            'RY | deg',
            x, y,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_rz = c.Label(
            'RZ | deg',
            x, y,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.group = c.Group(
            self.label_selected,
            self.label_type,
            self.label_position_x,
            self.label_position_y,
            self.label_position_z,
            self.label_size_w,
            self.label_size_h,
            self.label_size_d,
            self.label_rotation_rx,
            self.label_rotation_ry,
            self.label_rotation_rz,
        )
        self.update()

    def _anchor_all_labels(self):
        self.label_selected.left = self.base_x
        self.label_selected.top = self.base_y
        self.label_type.left = self.base_x + 3
        self.label_type.top = self.base_y + 16
        # position
        self.label_position_x.left = self.base_x + 5  # x
        self.label_position_x.top = self.base_y + 30
        self.label_position_y.left = self.base_x + 5   # y
        self.label_position_y.top = self.base_y + 40
        self.label_position_z.left = self.base_x + 5   # z
        self.label_position_z.top = self.base_y + 50
        # size
        self.label_size_w.left = self.base_x + 35   # w
        self.label_size_w.top = self.base_y + 30
        self.label_size_h.left = self.base_x + 35   # h
        self.label_size_h.top = self.base_y + 40
        self.label_size_d.left = self.base_x + 35   # d
        self.label_size_d.top = self.base_y + 50
        # rotation
        self.label_rotation_rx.left = self.base_x + 5   # rx
        self.label_rotation_rx.top = self.base_y + 70
        self.label_rotation_ry.left = self.base_x + 5   # ry
        self.label_rotation_ry.top = self.base_y + 80
        self.label_rotation_rz.left = self.base_x + 5   # rz
        self.label_rotation_rz.top = self.base_y + 90

    def update(self):
        obj = get_selected_object('obj')
        name = get_selected_object('name')
        obj_type = get_selected_object('type')
        self.label_selected.value = str(name)
        if obj is None:
            self.label_type.value = '-'
            self.label_position_x.value = '-'
            self.label_position_y.value = '-'
            self.label_position_z.value = '-'
            self.label_size_w.value = '-'
            self.label_size_h.value = '-'
            self.label_size_d.value = '-'
            self.label_rotation_rx.value = '-'
            self.label_rotation_ry.value = '-'
            self.label_rotation_rz.value = '-'
        else:
            self.label_type.value = str(obj_type) if obj_type is not None else '-'
            self.label_position_x.value = str(obj.cord.x)
            self.label_position_y.value = str(obj.cord.y)
            self.label_position_z.value = str(obj.cord.z)
            self.label_size_w.value = str(obj.size.x)
            self.label_size_h.value = str(obj.size.y)
            self.label_size_d.value = str(obj.size.z)
            self.label_rotation_rx.value = str(obj.rotation.x) + ' (' + str(round(obj.rotation.x * 180.0 / m.pi)) + 'º)'
            self.label_rotation_ry.value = str(obj.rotation.y) + ' (' + str(round(obj.rotation.y * 180.0 / m.pi)) + 'º)'
            self.label_rotation_rz.value = str(obj.rotation.z) + ' (' + str(round(obj.rotation.z * 180.0 / m.pi)) + 'º)'
        self._anchor_all_labels()


### GRAPHICS
drawAxis()
depth_layer = DepthLayerManager()
cuboid1 = Cuboid(Vec3(0, 0, 0), Vec3(50, 50, 50))
# cuboid2 = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))
# cuboid_which_is_named_very_long = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))

register_object(cuboid1, 'cuboid1')
# register_object(cuboid2, 'cuboid2')
# register_object(cuboid_which_is_named_very_long, 'cuboid_which_is_named_very_long')

selected_object_info = SelectedObjectInfo()
input_info = InputInfo()
camera_info = CameraInfo()


### EVENTS
def onKeyHold(keys, modifiers=None):
    input_info.set_keyboard(keys, modifiers)
    
    ## CAMERA
    cam_updated = False
    # translation
    cdx = (-5 if 'j' in keys else 0) + (+5 if 'l' in keys else 0)
    cdy = (+5 if 'i' in keys else 0) + (-5 if 'k' in keys else 0)
    cdz = (+5 if '.' in keys else 0) + (-5 if ',' in keys else 0)
    if cdx or cdy or cdz:
        camera.move(Vec3(cdx, cdy, cdz))
        cam_updated = True
    ## rotation
    cdrx = (m.pi/180.0) * ((+5 if 'h' in keys else 0) + (-5 if 'y' in keys else 0))
    cdry = (m.pi/180.0) * ((+5 if 'm' in keys else 0) + (-5 if 'n' in keys else 0))
    cdrz = (m.pi/180.0) * ((+5 if 'u' in keys else 0) + (-5 if 'o' in keys else 0))
    if cdrx or cdry or cdrz:
        camera.rotate(Vec3(cdrx, cdry, cdrz))
        cam_updated = True
    
    if cam_updated:
        camera_info.update()
    
    ## OBJECTS
    selected_group = get_selected_object('obj')
    if selected_group is None:
        return
    obj_updated = False
    # Translation
    dx = (-5 if 'a' in keys else 0) + (+5 if 'd' in keys else 0)
    dy = (+5 if 'w' in keys else 0) + (-5 if 's' in keys else 0)
    dz = (+5 if 'x' in keys else 0) + (-5 if 'z' in keys else 0)
    if dx or dy or dz:
        if hasattr(selected_group, 'move'):
            selected_group.move(Vec3(dx, dy, dz))
            obj_updated = True
    # Scale
    dw = (-5 if 'A' in keys else 0) + (+5 if 'D' in keys else 0)
    dh = (+5 if 'W' in keys else 0) + (-5 if 'S' in keys else 0)
    dd = (+5 if 'X' in keys else 0) + (-5 if 'Z' in keys else 0)
    if dw or dh or dd:
        if hasattr(selected_group, 'scale'):
            selected_group.scale(Vec3(dw, dh, dd))
            obj_updated = True
    # Rotation
    drx = (m.pi/180.0) * ((+5 if 'r' in keys else 0) + (-5 if 'f' in keys else 0))
    dry = (m.pi/180.0) * ((+5 if 'c' in keys else 0) + (-5 if 'v' in keys else 0))
    drz = (m.pi/180.0) * ((+5 if 'q' in keys else 0) + (-5 if 'e' in keys else 0))
    if drx or dry or drz:
        if hasattr(selected_group, 'rotate'):
            selected_group.rotate(Vec3(drx, dry, drz))
            obj_updated = True
    
    if obj_updated:
        selected_object_info.update()
        depth_layer.update(_objects, camera.cord, camera.rotation, _axis_group)

def onKeyPress(keys, modifiers=None):
    input_info.set_keyboard(keys, modifiers)

    if 'tab' in keys:
        select_next_object()
        selected_object_info.update()
    
    if 'backspace' in keys:
        clearObject(get_selected_object('name'))
        select_next_object()
        selected_object_info.update()
    
    if '+' in keys:
        new_obj = Cuboid(Vec3(0, 0, 0), Vec3(50, 50, 50))
        base_name = 'cuboid'
        index = 1
        new_name = f'{base_name}{index}'
        while new_name in _objects:
            index += 1
            new_name = f'{base_name}{index}'
        register_object(new_obj, new_name)
        select_next_object()
        selected_object_info.update()

    # if '/' in keys:
    #     command = command_input()
    
    if 'g' in keys:
        global SHOW_AXIS
        SHOW_AXIS = not SHOW_AXIS
        drawAxis()
    
    if 't' in keys:
        selected_group = get_selected_object('obj')
        if selected_group is None:
            return
        if hasattr(selected_group, 'reset_cord'):
            selected_group.reset_cord()
        if hasattr(selected_group, 'reset_size'):
            selected_group.reset_size()
        if hasattr(selected_group, 'reset_rotation'):
            selected_group.reset_rotation()
        selected_object_info.update()
    if 'b' in keys and 'tab' not in keys:   # WHY ALSO REACTS TO TAB?
        camera.reset()
        camera_info.update()

def onKeyRelease(keys=None, modifiers=None):
    input_info.set_keyboard(None, None)

def onMouseMove(x, y, button=None):
    input_info.set_mouse(x, y, button)

def onMouseDrag(x, y, button=None):
    input_info.set_mouse(x, y, button)
    input_info.set_mouse_pressed(True)

def onMousePress(x, y, button=None):
    input_info.set_mouse(x, y, button)
    input_info.set_mouse_pressed(True)

def onMouseRelease(x, y, button=None):
    input_info.set_mouse(x, y, None)
    input_info.set_mouse_pressed(False)

if CMU_RUN:
    c.run()