from __future__ import annotations
from dataclasses import dataclass
import math as m
from cmu_graphics import *  # general
from cmu_graphics import cmu_graphics as c  # c.run()

app.background = 'black'

CMU_RUN = True
f = 250
cam = (200, 200)


### define vector(1x3), matrix(3x3)
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


def drawAxis():
    axisVerts = [
        Vec3(-200, 0, 250), Vec3(+200, 0, 250), # x-axis
        Vec3(0, -200, 250), Vec3(0, +200, 250), # y-axis
        Vec3(0, 0, 1e-6), Vec3(0, 0, 1e+6)  # z-axis
    ]
    print(axisVerts)
    projectedAxisVerts = [projectToScreen(something, cam) for something in axisVerts]
    print(projectedAxisVerts)
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

def Cuboid(cord: Vec3, size: tuple[float, float, float], cam: tuple[float, float] = None):
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
        return
    
    try:
        cuboidGroup = Group(
            # Front face (vertices 4, 5, 6, 7)
            Polygon(p[4][0], p[4][1],
                    p[5][0], p[5][1],
                    p[6][0], p[6][1],
                    p[7][0], p[7][1],
                    fill='lightblue', border='black', borderWidth=1, opacity=50),
            # Back face (vertices 0, 1, 2, 3)
            Polygon(p[0][0], p[0][1],
                    p[1][0], p[1][1],
                    p[2][0], p[2][1],
                    p[3][0], p[3][1],
                    fill='lightblue', border='black', borderWidth=1, opacity=50),
            # Top face (vertices 3, 2, 6, 7)
            Polygon(p[3][0], p[3][1],
                    p[2][0], p[2][1],
                    p[6][0], p[6][1],
                    p[7][0], p[7][1],
                    fill='lightgreen', border='black', borderWidth=1, opacity=50),
            # Bottom face (vertices 0, 1, 5, 4)
            Polygon(p[0][0], p[0][1],
                    p[1][0], p[1][1],
                    p[5][0], p[5][1],
                    p[4][0], p[4][1],
                    fill='lightgreen', border='black', borderWidth=1, opacity=50),
            # Right face (vertices 1, 2, 6, 5)
            Polygon(p[1][0], p[1][1],
                    p[2][0], p[2][1],
                    p[6][0], p[6][1],
                    p[5][0], p[5][1],
                    fill='lightyellow', border='black', borderWidth=1, opacity=50),
            # Left face (vertices 0, 3, 7, 4)
            Polygon(p[0][0], p[0][1],
                    p[3][0], p[3][1],
                    p[7][0], p[7][1],
                    p[4][0], p[4][1],
                    fill='lightyellow', border='black', borderWidth=1, opacity=50)
        )
    except: pass
    return cuboidGroup


# def Cube(cord: Vec3, size: float, cam: tuple[float, float] = None):
#     Cuboid(cord, (size, size, size), cam)
#     return cord, size, cam


x = -50
y = -50
z = 150
size = 50.0

# a = Cuboid(Vec3(x, y, z), (size, size, size))

# print(a, type(a))



def onKeyHold(keys):
    global x, y, z, size

    if ('a' in keys):
        x += -10
        # a.cord += Vec3(-10, 0, 0)
    elif ('d' in keys):
        x += 10
        # a.cord += Vec3(+10, 0, 0)
    elif ('w' in keys):
        y += 10
    elif ('s' in keys):
        y += -10
    elif ('z' in keys):
        z += -10
    elif ('x' in keys):
        z += 10
    
    Rect(-200, -200, 800, 800, fill='white', border=None)
    drawAxis()
    Cuboid(Vec3(x, y, z), (size, size, size))






if CMU_RUN:
    c.run()