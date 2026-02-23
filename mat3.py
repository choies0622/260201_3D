from __future__ import annotations
from dataclasses import dataclass
import math as m
from cmu_graphics import cmu_graphics as c  # CMU GENERAL


CMU_RUN = True

f = 250

MIN_SIZE = 1e-6  # prevent size < 0
NEAR_PLANE = 1.0  # prevent extreme projections when z is near 0

def _clamp_size(v: float) -> float:
    return round(v, 5) if v > MIN_SIZE else MIN_SIZE
    # return round(max(MIN_SIZE, v), 5)
def _clamp_vec3_size(v: Vec3) -> Vec3:
    return Vec3(_clamp_size(v.x), _clamp_size(v.y), _clamp_size(v.z))

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
            x, y, z = v2s(other)
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

@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def norm(self) -> float:
        return m.sqrt(self.x * self.x + self.y * self.y)

@dataclass(frozen=True)
class Mat2:
    m: tuple[tuple[float, float],
             tuple[float, float]]

    def __matmul__(self, other):
        # Mat2 @ Vec2
        if isinstance(other, Vec2):
            return Vec2(
                self.m[0][0] * other.x + self.m[0][1] * other.y,
                self.m[1][0] * other.x + self.m[1][1] * other.y,
            )

        # Mat2 @ Mat2
        if isinstance(other, Mat2):
            A, B = self.m, other.m

            def cell(i, j):
                return A[i][0]*B[0][j] + A[i][1]*B[1][j]

            return Mat2((
                (cell(0, 0), cell(0, 1)),
                (cell(1, 0), cell(1, 1)),
            ))

        return NotImplemented

    def eigen(self) -> tuple[tuple[float, Vec2], tuple[float, Vec2]]:
        """Eigenvalues/eigenvectors of a 2x2 symmetric matrix.
        Returns ((lambda1, v1), (lambda2, v2)) with lambda1 >= lambda2."""
        a, b = self.m[0]
        _, d = self.m[1]
        tr = a + d
        det = a * d - b * b
        disc = m.sqrt(max(0.0, tr * tr - 4.0 * det))
        lam1 = (tr + disc) / 2.0
        lam2 = (tr - disc) / 2.0
        if abs(b) > 1e-12:
            v1 = Vec2(lam1 - d, b)
            v2 = Vec2(lam2 - d, b)
        elif abs(a - d) > 1e-12:
            v1 = Vec2(1.0, 0.0) if a >= d else Vec2(0.0, 1.0)
            v2 = Vec2(0.0, 1.0) if a >= d else Vec2(1.0, 0.0)
        else:
            v1 = Vec2(1.0, 0.0)
            v2 = Vec2(0.0, 1.0)
        len1 = v1.norm()
        len2 = v2.norm()
        if len1 > 1e-12:
            v1 = Vec2(v1.x / len1, v1.y / len1)
        if len2 > 1e-12:
            v2 = Vec2(v2.x / len2, v2.y / len2)
        return ((lam1, v1), (lam2, v2))

def v2s(v: Vec3) -> tuple[float, float, float]:
    return v.x, v.y, v.z
def s2v(s: float) -> Vec3:
    return Vec3(s, s, s)


### CAMERA
cam_cord = Vec3(0, 0, -200)
cam_rotation = s2v(0)

class Camera:
    """
    CAMARA (p, q, r)

    - TRANSLATION
        - move (delta: Vec3)
        - set_cord (cord: Vec3)
        - reset_cord
    - ROTATION
        - rotate (delta: Vec3)
        - set_rotation (rotation: Vec3)
        - reset_rotation
    """
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

    def reset_cord(self):
        self.cord = self._initial_cord
        self._sync_globals()
        self._sync_objects()
    
    def reset_rotation(self):
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
    rx, ry, rz = v2s(rotation)
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
    cam_rotation: Vec3 | None = None,
) -> Vec3:
    """
    WORLD COORD SYS: x, y, z  ->  CAMERA COORD SYS: p, q, r
    """
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    local = points - cam_cord
    if cam_rotation == s2v(0):
        return local
    rx, ry, rz = v2s(cam_rotation)
    # Inverse rotation to move world space into camera space
    R = Rx(-rx) @ Ry(-ry) @ Rz(-rz)
    return R @ local
def projectToScreen(
    points: Vec3,
    screen_center: tuple[float, float] | None = None,
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
    p0: Vec3, p1: Vec3, near: float = 1e-6,
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


### 3D PRIMITIVES
def Line(
    p0: Vec3, p1: Vec3,
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None,
    screen_center: tuple[float, float] | None = None,
    **kwargs,
) -> c.Line | None:
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    if screen_center is None:
        screen_center = globals()['screen_center']
    p0_cam = projectToCamera(p0, cam_cord, cam_rotation)
    p1_cam = projectToCamera(p1, cam_cord, cam_rotation)
    clipped = _clip_segment_to_near_plane(p0_cam, p1_cam)
    if clipped is None:
        return None
    q0 = projectToScreen(clipped[0], screen_center)
    q1 = projectToScreen(clipped[1], screen_center)
    if q0 is None or q1 is None:
        return None
    try:
        return c.Line(q0[0], q0[1], q1[0], q1[1], **kwargs)
    except:
        return None

def Polygon(
    verts: list[Vec3],
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None,
    screen_center: tuple[float, float] | None = None,
    **kwargs,
) -> c.Polygon | None:
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    if screen_center is None:
        screen_center = globals()['screen_center']
    points = []
    for v in verts:
        cam_v = projectToCamera(v, cam_cord, cam_rotation)
        if cam_v.z < NEAR_PLANE:
            return None
        scr = projectToScreen(cam_v, screen_center)
        if scr is None:
            return None
        points.append(scr)
    flat = []
    for u, v in points:
        flat.append(u)
        flat.append(v)
    try:
        return c.Polygon(*flat, **kwargs)
    except:
        return None

def _rotateToCam(v: Vec3, cam_rotation: Vec3) -> Vec3:
    """Rotate a direction vector from world to camera space (no translation)."""
    if cam_rotation == s2v(0):
        return v
    rx, ry, rz = v2s(cam_rotation)
    R = Rx(-rx) @ Ry(-ry) @ Rz(-rz)
    return R @ v

def _oval_params_from_cam(
    cord_cam: Vec3,
    s_cam: Vec3,
    t_cam: Vec3,
    screen_center: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, float] | None:
    """Camera-space oval data -> (u, v, width, height, angle_deg) or None."""
    if screen_center is None:
        screen_center = globals()['screen_center']
    if cord_cam.z < NEAR_PLANE:
        return None
    scr = projectToScreen(cord_cam, screen_center)
    if scr is None:
        return None
    u, v = scr
    k = f / cord_cam.z
    ds = Vec2(k * s_cam.x, -k * s_cam.y)
    dt = Vec2(k * t_cam.x, -k * t_cam.y)
    gram = Mat2((
        (ds.dot(ds), ds.dot(dt)),
        (ds.dot(dt), dt.dot(dt)),
    ))
    (lam1, v1), (lam2, _) = gram.eigen()
    semi_major = m.sqrt(max(0.0, lam1))
    semi_minor = m.sqrt(max(0.0, lam2))
    if semi_major < 1e-6:
        return None
    angle_deg = m.degrees(m.atan2(v1.y, v1.x))
    return (u, v, 2.0 * semi_major, 2.0 * semi_minor, angle_deg)

def Oval(
    cord: Vec3,
    s: Vec3,
    t: Vec3,
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None,
    screen_center: tuple[float, float] | None = None,
    **kwargs,
) -> c.Oval | None:
    if cam_cord is None:
        cam_cord = globals()['cam_cord']
    if cam_rotation is None:
        cam_rotation = globals()['cam_rotation']
    if screen_center is None:
        screen_center = globals()['screen_center']
    cord_cam = projectToCamera(cord, cam_cord, cam_rotation)
    s_cam = _rotateToCam(s, cam_rotation)
    t_cam = _rotateToCam(t, cam_rotation)
    params = _oval_params_from_cam(cord_cam, s_cam, t_cam, screen_center)
    if params is None:
        return None
    u, v, w, h, angle_deg = params
    try:
        return c.Oval(
            u, v, w, h,
            rotateAngle=angle_deg,
            **kwargs,
        )
    except:
        return None


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
        l = Line(p0_world, p1_world, fill=color, dashes=True, arrowEnd=True, opacity=50)
        if l is not None:
            _axis_group.add(l)
    return _axis_group

# Cuboid
def _cuboid_faces(
    cord: Vec3,
    size: Vec3,
    rotation: Vec3 | None = None,
    cam_cord: Vec3 | None = None,
    cam_rotation: Vec3 | None = None,
    screen_center: tuple[float, float] | None = None,
    fill: list[str, str, str, str, str, str] | None = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightyellow'],
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
        Vec3(+w, +h, +d), Vec3(-w, +h, +d),
    ]
    if rotation is not None and rotation != s2v(0):
        verts = rotateVerts(verts, rotation)
    verts = [v + cord for v in verts]

    face_indices = [
        [4, 5, 6, 7],  # front
        [0, 1, 2, 3],  # back
        [3, 2, 6, 7],  # top
        [0, 1, 5, 4],  # bottom
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4],  # left
    ]
    faces = []
    for idx, fi in enumerate(face_indices):
        face_verts = [verts[i] for i in fi]
        poly = Polygon(
            face_verts, cam_cord, cam_rotation, screen_center,
            fill=fill[idx], border='black', borderWidth=1, opacity=50,
        )
        if poly is None:
            return None
        faces.append(poly)
    return faces

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
        self.size = _clamp_vec3_size(size)
        self._initial_size = self.size
        self.rotation = rotation if rotation is not None else s2v(0)
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
            self.screen_center,
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
        self.size = _clamp_vec3_size(Vec3(
            self.size.x + delta.x,
            self.size.y + delta.y,
            self.size.z + delta.z,
        ))
        return self.redraw()

    def set_size(self, size: Vec3):
        self.size = _clamp_vec3_size(size)
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

# Cube
class Cube:
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
        size: float,
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
        s = _clamp_size(size)
        self.size = s2v(s)
        self._initial_size = self.size
        self.rotation = rotation if rotation is not None else s2v(0)
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
            self.screen_center,
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
        sigma = delta.x + delta.y + delta.z
        self.size = _clamp_vec3_size(Vec3(
            self.size.x + sigma,
            self.size.y + sigma,
            self.size.z + sigma,
        ))
        return self.redraw()
        
    def set_size(self, size: float | Vec3):
        if isinstance(size, float):
            s = _clamp_size(size)
            self.size = s2v(s)
        else:
            self.size = _clamp_vec3_size(size)
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

# Sphere
def _sphere_silhouette(
    cord_cam: Vec3,
    radius: float,
) -> tuple[Vec3, Vec3, Vec3] | None:
    """Returns (silhouette_center_cam, s_cam, t_cam) in camera space, or None."""
    d2 = cord_cam.x**2 + cord_cam.y**2 + cord_cam.z**2
    r2 = radius * radius
    if d2 <= r2:
        return None
    d = m.sqrt(d2)
    ratio = r2 / d2
    sil_center = Vec3(
        cord_cam.x * (1.0 - ratio),
        cord_cam.y * (1.0 - ratio),
        cord_cam.z * (1.0 - ratio),
    )
    r_s = radius * m.sqrt(d2 - r2) / d
    nx, ny, nz = cord_cam.x / d, cord_cam.y / d, cord_cam.z / d
    up = Vec3(1.0, 0.0, 0.0) if abs(nx) < 0.9 else Vec3(0.0, 1.0, 0.0)
    # t1 = normalize(cross(n, up))
    t1x = ny * up.z - nz * up.y
    t1y = nz * up.x - nx * up.z
    t1z = nx * up.y - ny * up.x
    t1_len = m.sqrt(t1x*t1x + t1y*t1y + t1z*t1z)
    if t1_len < 1e-12:
        return None
    t1x /= t1_len; t1y /= t1_len; t1z /= t1_len
    # t2 = cross(n, t1)
    t2x = ny * t1z - nz * t1y
    t2y = nz * t1x - nx * t1z
    t2z = nx * t1y - ny * t1x
    s_vec = Vec3(t1x * r_s, t1y * r_s, t1z * r_s)
    t_vec = Vec3(t2x * r_s, t2y * r_s, t2z * r_s)
    return (sil_center, s_vec, t_vec)

def _sphere_ovals(
    cord: Vec3, radius: float, rotation: Vec3,
    cam_cord: Vec3, cam_rotation: Vec3,
    screen_center: tuple[float, float],
) -> list | None:
    ovals = []
    gc_pairs = [
        (Vec3(radius, 0, 0), Vec3(0, radius, 0), 'blue'),   # XY plane (normal Z)
        (Vec3(radius, 0, 0), Vec3(0, 0, radius), 'green'),  # XZ plane (normal Y)
        (Vec3(0, radius, 0), Vec3(0, 0, radius), 'red'),    # YZ plane (normal X)
    ]
    for s_local, t_local, color in gc_pairs:
        if rotation != s2v(0):
            s_rot, t_rot = rotateVerts([s_local, t_local], rotation)
        else:
            s_rot, t_rot = s_local, t_local
        oval = Oval(
            cord, s_rot, t_rot,
            cam_cord, cam_rotation, screen_center,
            fill=None, border=color, borderWidth=1, opacity=50,
        )
        if oval is not None:
            ovals.append(oval)
    # Silhouette
    cord_cam = projectToCamera(cord, cam_cord, cam_rotation)
    sil = _sphere_silhouette(cord_cam, radius)
    if sil is not None:
        sil_center, sil_s, sil_t = sil
        params = _oval_params_from_cam(sil_center, sil_s, sil_t, screen_center)
        if params is not None:
            u, v, w, h, angle_deg = params
            try:
                sil_oval = c.Oval(
                    u, v, w, h,
                    rotateAngle=angle_deg,
                    fill='lightBlue', border='black',
                    borderWidth=1, opacity=30,
                )
                ovals.insert(0, sil_oval)
            except:
                pass
    return ovals if ovals else None

class Sphere:
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
        size: float,
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
        s = _clamp_size(size)
        self.size = s2v(s)
        self._initial_size = self.size
        self.rotation = rotation if rotation is not None else s2v(0)
        self._initial_rotation = self.rotation
        self.cam_cord = cam_cord
        self.cam_rotation = cam_rotation
        self.screen_center = screen_center
        self.group = c.Group()
        self.redraw()

    def redraw(self):
        ovals = _sphere_ovals(
            self.cord, self.size.x / 2.0, self.rotation,
            self.cam_cord, self.cam_rotation, self.screen_center,
        )
        self.group.clear()
        if ovals is None:
            self.group.visible = False
            return self.group
        self.group.visible = True
        for oval in ovals:
            self.group.add(oval)
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
    
    # Scale (uniform)
    def scale(self, delta: Vec3):
        sigma = delta.x + delta.y + delta.z
        self.size = _clamp_vec3_size(Vec3(
            self.size.x + sigma,
            self.size.y + sigma,
            self.size.z + sigma,
        ))
        return self.redraw()
        
    def set_size(self, size: float | Vec3):
        if isinstance(size, float):
            s = _clamp_size(size)
            self.size = s2v(s)
        else:
            self.size = _clamp_vec3_size(size)
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
    # c.Arc(x, y, w, h, startAngle, sweepAngle, visible=True)
    # c.Oval(x, y, w, h, rotateAngle=0, align='center', visible=True)
    # c.Circle(x, y, r, rotateAngle=0, visible=True)
# Cylinder
# Cone
# Pyramid


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
        self.label_camera_position_x.right = self.base_x - 15 # x
        self.label_camera_position_x.top = self.base_y - 60
        self.label_camera_position_y.right = self.base_x - 15 # y
        self.label_camera_position_y.top = self.base_y - 50
        self.label_camera_position_z.right = self.base_x - 15 # z
        self.label_camera_position_z.top = self.base_y - 40
        # rotation
        self.label_camera_rotation_rx.right = self.base_x - 5 # rx
        self.label_camera_rotation_rx.top = self.base_y - 20
        self.label_camera_rotation_ry.right = self.base_x - 5 # ry
        self.label_camera_rotation_ry.top = self.base_y - 10
        self.label_camera_rotation_rz.right = self.base_x - 5 # rz
        self.label_camera_rotation_rz.top = self.base_y
    def update(self):
        self.label_camera_position_x.value = str(float(camera.cord.x))
        self.label_camera_position_y.value = str(float(camera.cord.y))
        self.label_camera_position_z.value = str(float(camera.cord.z))
        self.label_camera_rotation_rx.value = str(float(camera.rotation.x)) + '  | ' + str(float(round(camera.rotation.x * 180.0 / m.pi, 2))) + 'º'
        self.label_camera_rotation_ry.value = str(float(camera.rotation.y)) + '  | ' + str(float(round(camera.rotation.y * 180.0 / m.pi, 2))) + 'º'
        self.label_camera_rotation_rz.value = str(float(camera.rotation.z)) + '  | ' + str(float(round(camera.rotation.z * 180.0 / m.pi, 2))) + 'º'
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
        self.label_size_w.left = self.base_x + 45   # w
        self.label_size_w.top = self.base_y + 30
        self.label_size_h.left = self.base_x + 45   # h
        self.label_size_h.top = self.base_y + 40
        self.label_size_d.left = self.base_x + 45   # d
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
            self.label_position_x.value = str(float(obj.cord.x))
            self.label_position_y.value = str(float(obj.cord.y))
            self.label_position_z.value = str(float(obj.cord.z))
            self.label_size_w.value = str(float(obj.size.x))
            self.label_size_h.value = str(float(obj.size.y))
            self.label_size_d.value = str(float(obj.size.z))
            self.label_rotation_rx.value = str(float(obj.rotation.x)) + '  | ' + str(float(round(obj.rotation.x * 180.0 / m.pi, 2))) + 'º'
            self.label_rotation_ry.value = str(float(obj.rotation.y)) + '  | ' + str(float(round(obj.rotation.y * 180.0 / m.pi, 2))) + 'º'
            self.label_rotation_rz.value = str(float(obj.rotation.z)) + '  | ' + str(float(round(obj.rotation.z * 180.0 / m.pi, 2))) + 'º'
        self._anchor_all_labels()


### GRAPHICS
drawAxis()
depth_layer = DepthLayerManager()
cuboid1 = Cuboid(s2v(0), s2v(50))
cube1 = Cube(s2v(0), 50)
sphere1 = Sphere(Vec3(0, 0, 0), 50)
# cuboid2 = Cuboid(Vec3(0, 0, 200), s2v(50))
# cuboid_which_is_named_very_long = Cuboid(Vec3(0, 0, 200), s2v(50))

register_object(cuboid1, 'cuboid1')
register_object(cube1, 'cube1')
register_object(sphere1, 'sphere1')
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
    else:   # WHY ALSO REACTS TO TAB?
        if 'backspace' in keys:
            clearObject(get_selected_object('name'))
            select_next_object()
            selected_object_info.update()
        
        if '+' in keys:
            new_obj = Cuboid(s2v(0), s2v(50))
            base_name = 'cuboid'
            index = 1
            new_name = f'{base_name}{index}'
            while new_name in _objects:
                index += 1
                new_name = f'{base_name}{index}'
            register_object(new_obj, new_name)
            select_next_object()
            selected_object_info.update()
        
        # AXIS
        if 'g' in keys:
            global SHOW_AXIS
            SHOW_AXIS = not SHOW_AXIS
            drawAxis()
        
        # RESET
        # Object
        if 't' in keys:
            selected_group = get_selected_object('obj')
            if selected_group is None:
                return
            if hasattr(selected_group, 'reset_cord'):
                selected_group.reset_cord()
            if hasattr(selected_group, 'reset_size'):
                selected_group.reset_size()
            selected_object_info.update()
        if 'T' in keys:
            selected_group = get_selected_object('obj')
            if selected_group is None:
                return
            if hasattr(selected_group, 'reset_rotation'):
                selected_group.reset_rotation()
            selected_object_info.update()
        
        # Camera
        if 'b' in keys:
            camera.reset_cord()
            camera_info.update()
        if 'B' in keys:
            camera.reset_rotation()
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