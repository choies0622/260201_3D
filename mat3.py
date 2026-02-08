from __future__ import annotations
from dataclasses import dataclass
import math as m
from cmu_graphics import *  # cmu general
from cmu_graphics import cmu_graphics as c  # c.run()


bg = Rect(-200, -200, 800, 800, fill='white', border=None)

CMU_RUN = True
SHOW_AXIS = True
f = 250
cam = (200, 200)


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


def projectToScreen(p: Vec3, cam: tuple[float, float]) -> tuple[float, float] | None:
    # camera looks along +z and p.z must be > 0
    # u = cx + f * (x / z)
    # v = cy - f * (y / z)
    if p.z <= 0:
        return None  # Behind camera or at camera plane
    cx, cy = cam
    u = cx + f * (p.x / p.z)
    v = cy - f * (p.y / p.z)
    return (u, v)


### MANAGEMENT

_objects: dict[str, object] = {}
_selected_index = 0

def register_object(obj: object, name: str):
    _objects[name] = obj

def clearObjects(name: str):
    global _selected_index
    obj = _objects.pop(name, None)
    if obj is not None:
        if isinstance(obj, Group):
            obj.clear()
        else:
            group = getattr(obj, 'group', None)
            if group is not None and hasattr(group, 'clear'):
                group.clear()
            elif hasattr(obj, 'visible'):
                obj.visible = False
    if _selected_index >= len(_objects):
        _selected_index = 0

def get_selected_object():
    if not _objects:
        return None
    return list(_objects.values())[_selected_index]

def get_selected_object_name():
    if not _objects:
        return None
    return list(_objects.keys())[_selected_index]


def select_next_object():
    global _selected_index
    if _objects:
        _selected_index = (_selected_index + 1) % len(_objects)


### OBJECTS
# Axis
def drawAxis():
    if not SHOW_AXIS:
        return None
    
    axisVerts = [
        Vec3(-200, 0, 250), Vec3(+200, 0, 250), # x-axis
        Vec3(0, -200, 250), Vec3(0, +200, 250), # y-axis
        Vec3(0, 0, 1e-6),   Vec3(0, 0, 1e+6)    # z-axis
    ]
    projectedAxisVerts = [projectToScreen(something, cam) for something in axisVerts]

    axis = Group(
        Line(
            projectedAxisVerts[0][0], projectedAxisVerts[0][1],
            projectedAxisVerts[1][0], projectedAxisVerts[1][1],
            fill = 'red', dashes = True, arrowEnd = True, opacity = 50
        ),
        Line(
            projectedAxisVerts[2][0], projectedAxisVerts[2][1],
            projectedAxisVerts[3][0], projectedAxisVerts[3][1],
            fill = 'green', dashes = True, arrowEnd = True, opacity = 50
        ),
        Line(
            projectedAxisVerts[4][0], projectedAxisVerts[4][1],
            projectedAxisVerts[5][0], projectedAxisVerts[5][1] - 1,
            fill = 'blue', dashes = True, arrowEnd = True, opacity = 50
        )
    )
    return axis

# Cuboid
def _cuboid_faces(
    cord: Vec3,
    size: Vec3,
    rotation: Vec3 | None = None,
    cam: tuple[float, float] | None = None,
    fill: list[str, str, str, str, str, str] | None = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightyellow'],
) -> list[Polygon] | None:
    if cam is None:
        cam = globals()['cam']

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
    
    p = [projectToScreen(v, cam) for v in verts]
    
    # Ignore vert beyond cam
    if any(coord is None for coord in p):
        return None
    
    try:
        # Front face (vertices 4, 5, 6, 7)
        front_face = Polygon(p[4][0], p[4][1],
                p[5][0], p[5][1],
                p[6][0], p[6][1],
                p[7][0], p[7][1],
                fill=fill[0], border='black', borderWidth=1, opacity=50)
        # Back face (vertices 0, 1, 2, 3)
        back_face = Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[3][0], p[3][1],
                fill=fill[1], border='black', borderWidth=1, opacity=50)
        # Top face (vertices 3, 2, 6, 7)
        top_face = Polygon(p[3][0], p[3][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[7][0], p[7][1],
                fill=fill[2], border='black', borderWidth=1, opacity=50)
        # Bottom face (vertices 0, 1, 5, 4)
        bottom_face = Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[5][0], p[5][1],
                p[4][0], p[4][1],
                fill=fill[3], border='black', borderWidth=1, opacity=50)
        # Right face (vertices 1, 2, 6, 5)
        right_face = Polygon(p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[5][0], p[5][1],
                fill=fill[4], border='black', borderWidth=1, opacity=50)
        # Left face (vertices 0, 3, 7, 4)
        left_face = Polygon(p[0][0], p[0][1],
                p[3][0], p[3][1],
                p[7][0], p[7][1],
                p[4][0], p[4][1],
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
    def __init__(
        self,
        cord: Vec3,
        size: Vec3,
        rotation: Vec3 | None = None,
        cam: tuple[float, float] | None = None,
    ):
        if cam is None:
            cam = globals()['cam']
        self.cord = cord
        self._initial_cord = cord
        self.size = size
        self._initial_size = size
        self.rotation = rotation if rotation is not None else Vec3(0, 0, 0)
        self._initial_rotation = self.rotation
        self.cam = cam
        self.group = Group()
        self.redraw()

    def redraw(self):
        faces = _cuboid_faces(self.cord, self.size, self.rotation, self.cam)
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
    def set_rotation(self, rotation: Vec3):
        self.rotation = rotation
        return self.redraw()

    def rotate(self, delta: Vec3):
        self.rotation = Vec3(
            self.rotation.x + delta.x,
            self.rotation.y + delta.y,
            self.rotation.z + delta.z,
        )
        return self.redraw()

    def reset_rotation(self):
        self.rotation = self._initial_rotation
        return self.redraw()


### UI

class SelectedObjectInfo:
    def __init__(self, x: float, y: float):
        self.base_x = x
        self.base_y = y
        self.label_selected = Label(
            'SELECTED NAME',
            x, y,
            fill='black', align='left-top', bold=True,
            size=15,
            opacity=80,
        )
        self.label_position_header = Label(
            'Position: ',
            x, y + 20,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_position_x = Label(
            'X',
            x + 50, y + 20,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_y = Label(
            'Y',
            x + 50, y + 30,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_z = Label(
            'Z',
            x + 50, y + 40,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_header = Label(
            'Size: ',
            x, y + 60,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_size_w = Label(
            'W',
            x + 50, y + 60,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_h = Label(
            'H',
            x + 50, y + 70,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_d = Label(
            'D',
            x + 50, y + 80,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_header = Label(
            'Rotation: ',
            x, y + 100,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_rotation_rx = Label(
            'RX',
            x + 50, y + 100,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_ry = Label(
            'RY',
            x + 50, y + 110,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_rz = Label(
            'RZ',
            x + 50, y + 120,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.group = Group(
            self.label_selected,
            self.label_position_header,
            self.label_position_x,
            self.label_position_y,
            self.label_position_z,
            self.label_size_header,
            self.label_size_w,
            self.label_size_h,
            self.label_size_d,
            self.label_rotation_header,
            self.label_rotation_rx,
            self.label_rotation_ry,
            self.label_rotation_rz,
        )
        self.update()

    def _anchor_label_selected(self):
        self.label_selected.left = self.base_x
        self.label_selected.top = self.base_y

    def update(self):
        obj = get_selected_object()
        name = get_selected_object_name()
        self.label_selected.value = str(name)
        self._anchor_label_selected()
        if obj is None:
            self.label_position_x.value = '-'
            self.label_position_y.value = '-'
            self.label_position_z.value = '-'
            self.label_size_w.value = '-'
            self.label_size_h.value = '-'
            self.label_size_d.value = '-'
            self.label_rotation_rx.value = '-'
            self.label_rotation_ry.value = '-'
            self.label_rotation_rz.value = '-'
            return
        self.label_position_x.value = str(obj.cord.x)
        self.label_position_y.value = str(obj.cord.y)
        self.label_position_z.value = str(obj.cord.z)
        self.label_size_w.value = str(obj.size.x)
        self.label_size_h.value = str(obj.size.y)
        self.label_size_d.value = str(obj.size.z)
        self.label_rotation_rx.value = str(obj.rotation.x)
        self.label_rotation_ry.value = str(obj.rotation.y)
        self.label_rotation_rz.value = str(obj.rotation.z)


### GRAPHICS

drawAxis()
cuboid1 = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))
# cuboid2 = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))
# cuboid_which_is_named_very_long = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))

register_object(cuboid1, 'cuboid1')
# register_object(cuboid2, 'cuboid2')
# register_object(cuboid_which_is_named_very_long, 'cuboid_which_is_named_very_long')

selected_object_info = SelectedObjectInfo(2, 2)


### EVENTS

def onKeyHold(keys):
    selected_group = get_selected_object()
    if selected_group is None:
        return
    updated = False

    # Translation
    dx = (-5 if 'a' in keys else 0) + (+5 if 'd' in keys else 0)
    dy = (+5 if 'w' in keys else 0) + (-5 if 's' in keys else 0)
    dz = (+5 if 'z' in keys else 0) + (-5 if 'x' in keys else 0)
    if dx or dy or dz:
        if hasattr(selected_group, 'move'):
            selected_group.move(Vec3(dx, dy, dz))
            updated = True
    
    # Scale
    dw = (-5 if 'A' in keys else 0) + (+5 if 'D' in keys else 0)
    dh = (+5 if 'W' in keys else 0) + (-5 if 'S' in keys else 0)
    dd = (+5 if 'Z' in keys else 0) + (-5 if 'X' in keys else 0)
    if dw or dh or dd:
        if hasattr(selected_group, 'scale'):
            selected_group.scale(Vec3(dw, dh, dd))
            updated = True
    
    # Rotation
    drx = (m.pi/180.0) * ((+5 if 'R' in keys else 0) + (-5 if 'F' in keys else 0))
    dry = (m.pi/180.0) * ((+5 if 'C' in keys else 0) + (-5 if 'V' in keys else 0))
    drz = (m.pi/180.0) * ((+5 if 'Q' in keys else 0) + (-5 if 'E' in keys else 0))
    if drx or dry or drz:
        if hasattr(selected_group, 'rotate'):
            selected_group.rotate(Vec3(drx, dry, drz))
            updated = True
    
    if updated:
        selected_object_info.update()

def onKeyPress(keys):
    if 'tab' in keys:
        select_next_object()
        selected_object_info.update()
    
    if '`' in keys:
        selected_group = get_selected_object()
        if selected_group is None:
            return
        if hasattr(selected_group, 'reset_cord'):
            selected_group.reset_cord()
        if hasattr(selected_group, 'reset_size'):
            selected_group.reset_size()
        if hasattr(selected_group, 'reset_rotation'):
            selected_group.reset_rotation()
        selected_object_info.update()


if CMU_RUN:
    c.run()