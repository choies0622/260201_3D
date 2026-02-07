from __future__ import annotations
from dataclasses import dataclass
import math as m
from cmu_graphics import *  # general
from cmu_graphics import cmu_graphics as c  # c.run()


bg = Rect(-200, -200, 800, 800, fill='white', border=None)

SHOW_AXIS = True
CMU_RUN = True
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


def rotateVerts(verts: list[Vec3], thetas: tuple[float, float, float]) -> list[Vec3]:
    thetaX, thetaY, thetaZ = thetas
    R = Rz(thetaZ) @ Ry(thetaY) @ Rx(thetaX)    # Order matters
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


### OBJECT
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
def _cuboid_faces(cord: Vec3, size: tuple[float, float, float], cam: tuple[float, float]) -> list[Polygon] | None:
    if cam is None:
        cam = globals()['cam']

    w, h, d = size
    w = w / 2.0; h = h / 2.0; d = d / 2.0; 
    verts = [
        Vec3(-w, -h, -d), Vec3(+w, -h, -d),
        Vec3(+w, +h, -d), Vec3(-w, +h, -d),
        Vec3(-w, -h, +d), Vec3(+w, -h, +d),
        Vec3(+w, +h, +d), Vec3(-w, +h, +d)
    ]
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
                fill='lightblue', border='black', borderWidth=1, opacity=50)
        # Back face (vertices 0, 1, 2, 3)
        back_face = Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[3][0], p[3][1],
                fill='lightblue', border='black', borderWidth=1, opacity=50)
        # Top face (vertices 3, 2, 6, 7)
        top_face = Polygon(p[3][0], p[3][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[7][0], p[7][1],
                fill='lightgreen', border='black', borderWidth=1, opacity=50)
        # Bottom face (vertices 0, 1, 5, 4)
        bottom_face = Polygon(p[0][0], p[0][1],
                p[1][0], p[1][1],
                p[5][0], p[5][1],
                p[4][0], p[4][1],
                fill='lightgreen', border='black', borderWidth=1, opacity=50)
        # Right face (vertices 1, 2, 6, 5)
        right_face = Polygon(p[1][0], p[1][1],
                p[2][0], p[2][1],
                p[6][0], p[6][1],
                p[5][0], p[5][1],
                fill='lightyellow', border='black', borderWidth=1, opacity=50)
        # Left face (vertices 0, 3, 7, 4)
        left_face = Polygon(p[0][0], p[0][1],
                p[3][0], p[3][1],
                p[7][0], p[7][1],
                p[4][0], p[4][1],
                fill='lightyellow', border='black', borderWidth=1, opacity=50)
    
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


def CuboidGroup(cord: Vec3, size: tuple[float, float, float], cam: tuple[float, float] = None):
    faces = _cuboid_faces(cord, size, cam)
    if faces is None:
        return None
    return Group(*faces)


class Cuboid:
    def __init__(self, cord: Vec3, size: tuple[float, float, float], cam: tuple[float, float] | None = None):
        if cam is None:
            cam = globals()['cam']
        self.cord = cord
        self.size = size
        self.cam = cam
        self.group = Group()
        self.redraw()

    def redraw(self):
        faces = _cuboid_faces(self.cord, self.size, self.cam)
        self.group.clear()
        if faces is None:
            self.group.visible = False
            return self.group
        self.group.visible = True
        for face in faces:
            self.group.add(face)
        return self.group

    def move(self, delta: Vec3):
        self.cord = self.cord + delta
        return self.redraw()

    def set_cord(self, cord: Vec3):
        self.cord = cord
        return self.redraw()

    def set_size(self, size: tuple[float, float, float]):
        self.size = size
        return self.redraw()


a = Cuboid(Vec3(0, 0, 200), (50, 50, 50))
a.move(Vec3(100, 0, 0))
a.set_size((70, 50, 50))

##### NEXT: graphics group management #####



if CMU_RUN:
    c.run()