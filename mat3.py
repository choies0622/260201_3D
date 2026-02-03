from __future__ import annotations
from dataclasses import dataclass
import math as m
from cmu_graphics import cmu_graphics as c

CMU_RUN = True
f = 0.8

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


def projectToScreen(p: Vec3, cam: tuple[float, float]) -> tuple[float, float]:
    # camera looks along +z and p.z must be > 0
    # u = cx + f * (x / z)
    # v = cy - f * (y / z)
    cx, cy = cam
    z = p.z if p.z != 0 else 1e-6   # Avoid division by zero or negative z (behind camera)
    u = cx + f * (p.x / z)
    v = cy - f * (p.y / z)
    return (u, v)

def rotateVerts(verts: list[Vec3], thetas: tuple[float, float, float]) -> list[Vec3]:
    thetaX, thetaY, thetaZ = thetas
    R = Rz(thetaZ) @ Ry(thetaY) @ Rx(thetaX) # Order matters
    return [R @ v for v in verts]


def Cuboid(cord: Vec3, size: tuple[float, float, float]) -> tuple[list[Vec3]]:
    w, h, d = size
    w /= 2.0; h /= 2.0; d /= 2.0; 
    verts = [
        Vec3(-w, -h, -d), Vec3(+w, -h, -d),
        Vec3(+w, +h, -d), Vec3(-w, +h, -d),
        Vec3(-w, -h, +d), Vec3(+w, -h, +d),
        Vec3(+w, +h, +d), Vec3(-w, +h, +d)
    ]
    verts = [v + cord for v in verts]
    # edges = [
    #     (0, 1), (1, 2), (2, 3), (3, 0), # back face
    #     (4, 5), (5, 6), (6, 7), (7, 4), # front face
    #     (0, 4), (1, 5), (2, 6), (3, 7)  # connecting edges
    # ]

    #### WORK ON IT ####
    projectToScreen()

    # c.Polygon(verts[0].0, y1, x2, y2, x3, y3, …, fill='black', border=None, borderWidth=2, opacity=100, rotateAngle=0, dashes=False, visible=True)

    return verts


def Cube(cord: Vec3, size: float) -> tuple[list[Vec3], list[tuple[int, int]]]:
    return Cuboid(cord, (size, size, size))











if CMU_RUN:
    c.run()