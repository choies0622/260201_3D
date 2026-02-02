from __future__ import annotations
from dataclasses import dataclass
import math as m


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


def project_perspective(p: Vec3, f: float, cx: float, cy: float) -> tuple[float, float]:
    """
    Perspective projection.
    - camera looks along +z and p.z must be > 0.
    - u = cx + f * (x / z)
    - v = cy - f * (y / z)
    """
    z = p.z if p.z != 0 else 1e-6   # Avoid division by zero or negative z (behind camera)
    u = cx + f * (p.x / z)
    v = cy - f * (p.y / z)
    return (u, v)


def make_cube(size: float) -> tuple[list[Vec3], list[tuple[int, int]]]:
    """
    Cube centered at origin, side length = size.
    Returns:
    - vertices: 8 points
    - edges: 12 pairs of vertex indices
    """
    s = size / 2.0
    verts = [
        Vec3(-s, -s, -s), Vec3(+s, -s, -s),
        Vec3(+s, +s, -s), Vec3(-s, +s, -s),
        Vec3(-s, -s, +s), Vec3(+s, -s, +s),
        Vec3(+s, +s, +s), Vec3(-s, +s, +s),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # back face
        (4, 5), (5, 6), (6, 7), (7, 4),  # front face
        (0, 4), (1, 5), (2, 6), (3, 7),  # connections
    ]
    return verts, edges
