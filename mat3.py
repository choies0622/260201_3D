from __future__ import annotations
from dataclasses import dataclass
import math as m
# from cmu_graphics import *  # cmu general
from cmu_graphics import cmu_graphics as c  # c.run()


bg = c.Rect(-200, -200, 800, 800, fill='white', border=None)

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


### PROJECTION
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
_objects: dict[str, dict[str, object]] = {}
_selected_index = 0

def register_object(obj: object, name: str, obj_type: str | None = None):
    if obj_type is None:
        obj_type = type(obj).__name__
    _objects[name] = {
        'obj': obj,
        'type': obj_type,
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
def drawAxis():
    if not SHOW_AXIS:
        return None
    
    axisVerts = [
        Vec3(-200, 0, 250), Vec3(+200, 0, 250), # x-axis
        Vec3(0, -200, 250), Vec3(0, +200, 250), # y-axis
        Vec3(0, 0, 1e-6),   Vec3(0, 0, 1e+6)    # z-axis
    ]
    projectedAxisVerts = [projectToScreen(something, cam) for something in axisVerts]

    axis = c.Group(
        c.Line(
            projectedAxisVerts[0][0], projectedAxisVerts[0][1],
            projectedAxisVerts[1][0], projectedAxisVerts[1][1],
            fill = 'red', dashes = True, arrowEnd = True, opacity = 50
        ),
        c.Line(
            projectedAxisVerts[2][0], projectedAxisVerts[2][1],
            projectedAxisVerts[3][0], projectedAxisVerts[3][1],
            fill = 'green', dashes = True, arrowEnd = True, opacity = 50
        ),
        c.Line(
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
    fill: list[str, str, str, str, str, str] | None = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightyellow', 'lightyellow']
) -> list[c.Polygon] | None:
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
    
    center_p = projectToScreen(cord, cam)
    if center_p is None:
        return None
    
    try:
        # Front face (vertices 4, 5, 6, 7)
        front_face = c.Polygon(p[4][0], p[4][1],
                p[5][0], p[5][1],
                p[6][0], p[6][1],
                p[7][0], p[7][1],
                fill=fill[0], border='black', borderWidth=1, opacity=50)
        # Back face (vertices 0, 1, 2, 3)
        back_face = c.Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[3][0], p[3][1],
                fill=fill[1], border='black', borderWidth=1, opacity=50)
        # Top face (vertices 3, 2, 6, 7)
        top_face = c.Polygon(p[3][0], p[3][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[7][0], p[7][1],
                fill=fill[2], border='black', borderWidth=1, opacity=50)
        # Bottom face (vertices 0, 1, 5, 4)
        bottom_face = c.Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[5][0], p[5][1],
                p[4][0], p[4][1],
                fill=fill[3], border='black', borderWidth=1, opacity=50)
        # Right face (vertices 1, 2, 6, 5)
        right_face = c.Polygon(p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[5][0], p[5][1],
                fill=fill[4], border='black', borderWidth=1, opacity=50)
        # Left face (vertices 0, 3, 7, 4)
        left_face = c.Polygon(p[0][0], p[0][1],
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
        self.group = c.Group()
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


### UI
def ask(prompt: str) -> str:
    return c.app.getTextInput(prompt)
def alert(message: str | None):
    if message is None:
        return
    c.app.showMessage(message)

def command_input():
    command = ask('Command:')
    alert(command)
    return command

class InputInfo:
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

class SelectedObjectInfo:
    def __init__(self, x: float, y: float):
        self.base_x = x
        self.base_y = y
        self.label_selected = c.Label(
            'SELECTED NAME',
            x, y,
            fill='black', align='left-top', bold=True,
            size=15,
            opacity=80,
        )
        self.label_type_header = c.Label(
            'Type: ',
            x, y + 25,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_type_value = c.Label(
            '-',
            x + 50, y + 25,
            fill='black', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_header = c.Label(
            'Position: ',
            x, y + 40,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_position_x = c.Label(
            'X',
            x + 50, y + 40,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_y = c.Label(
            'Y',
            x + 50, y + 50,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_position_z = c.Label(
            'Z',
            x + 50, y + 60,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_header = c.Label(
            'Size: ',
            x, y + 80,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_size_w = c.Label(
            'W',
            x + 50, y + 80,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_h = c.Label(
            'H',
            x + 50, y + 90,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_size_d = c.Label(
            'D',
            x + 50, y + 100,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_header = c.Label(
            'Rotation: ',
            x, y + 120,
            fill='black', align='left',
            size=10,
            opacity=80,
        )
        self.label_rotation_rx = c.Label(
            'RX',
            x + 50, y + 120,
            fill='red', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_ry = c.Label(
            'RY',
            x + 50, y + 130,
            fill='green', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.label_rotation_rz = c.Label(
            'RZ',
            x + 50, y + 140,
            fill='blue', align='left', bold=True,
            size=10,
            font='monospace',
            opacity=80,
        )
        self.group = c.Group(
            self.label_selected,
            self.label_type_header,
            self.label_type_value,
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
        obj = get_selected_object('obj')
        name = get_selected_object('name')
        obj_type = get_selected_object('type')
        self.label_selected.value = str(name)
        self._anchor_label_selected()
        if obj is None:
            self.label_type_value.value = '-'
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
        self.label_type_value.value = str(obj_type) if obj_type is not None else '-'
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
input_info = InputInfo()


### EVENTS
def onKeyHold(keys, modifiers=None):
    input_info.set_keyboard(keys, modifiers)
    
    selected_group = get_selected_object('obj')
    if selected_group is None:
        return
    updated = False

    # Translation
    dx = (-5 if 'a' in keys else 0) + (+5 if 'd' in keys else 0)
    dy = (+5 if 'w' in keys else 0) + (-5 if 's' in keys else 0)
    dz = (+5 if 'x' in keys else 0) + (-5 if 'z' in keys else 0)
    if dx or dy or dz:
        if hasattr(selected_group, 'move'):
            selected_group.move(Vec3(dx, dy, dz))
            updated = True
    
    # Scale
    dw = (-5 if 'A' in keys else 0) + (+5 if 'D' in keys else 0)
    dh = (+5 if 'W' in keys else 0) + (-5 if 'S' in keys else 0)
    dd = (+5 if 'X' in keys else 0) + (-5 if 'Z' in keys else 0)
    if dw or dh or dd:
        if hasattr(selected_group, 'scale'):
            selected_group.scale(Vec3(dw, dh, dd))
            updated = True
    
    # Rotation
    drx = (m.pi/180.0) * ((+5 if 'r' in keys else 0) + (-5 if 'f' in keys else 0))
    dry = (m.pi/180.0) * ((+5 if 'c' in keys else 0) + (-5 if 'v' in keys else 0))
    drz = (m.pi/180.0) * ((+5 if 'q' in keys else 0) + (-5 if 'e' in keys else 0))
    if drx or dry or drz:
        if hasattr(selected_group, 'rotate'):
            selected_group.rotate(Vec3(drx, dry, drz))
            updated = True
    
    if updated:
        selected_object_info.update()

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
        new_obj = Cuboid(Vec3(0, 0, 200), Vec3(50, 50, 50))
        base_name = 'cuboid'
        index = 1
        new_name = f'{base_name}{index}'
        while new_name in _objects:
            index += 1
            new_name = f'{base_name}{index}'
        register_object(new_obj, new_name)
        select_next_object()
        selected_object_info.update()

    if '/' in keys:
        command = command_input()
    
    if '`' in keys:
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