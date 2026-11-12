"""Ray-triangle intersection: is a seat in shade or in direct sun?

Idea: from the seat point p, draw a ray towards the sun, p + λ·s (λ > 0).
If the ray hits any roof, wall or stand triangle, the seat is in SHADE; otherwise it is in
direct SUN.

Method: Möller & Trumbore (1997). Solve  p + λ·s = V0 + u·E1 + v·E2  for (λ, u, v).
The ray hits the triangle when  u ≥ 0, v ≥ 0, u + v ≤ 1  and  λ > 0.

Two versions of the same test live here:
  ray_hits_triangle / is_shaded   one ray, one triangle; easy to read and to check by hand
  mt_all / first_hit_t            all points × all times × all triangles at once (numpy)
tests/test_raytrace.py checks that they agree on every point and time step.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9          # |determinant| below this: ray parallel to the triangle's plane
MIN_LAMBDA = 1e-6   # hits closer than this are the seat's own surface, not an obstacle


# ---------------------------------------------------------------------------
# Scalar version
# ---------------------------------------------------------------------------
def ray_hits_triangle(p, s, V0, V1, V2):
    """Return (hit, λ, u, v) for one ray (p, s) and one triangle (V0, V1, V2)."""
    p, s, V0, V1, V2 = (np.asarray(x, dtype=float) for x in (p, s, V0, V1, V2))
    E1 = V1 - V0                     # triangle edge 1
    E2 = V2 - V0                     # triangle edge 2
    h = np.cross(s, E2)
    a = np.dot(E1, h)                # determinant
    if abs(a) < EPS:                 # ray parallel to the triangle's plane
        return False, None, None, None
    f = 1.0 / a
    w = p - V0
    u = f * np.dot(w, h)             # 1st barycentric coordinate
    if u < 0 or u > 1:
        return False, None, u, None
    q = np.cross(w, E1)
    v = f * np.dot(s, q)             # 2nd barycentric coordinate
    if v < 0 or u + v > 1:
        return False, None, u, v
    lam = f * np.dot(E2, q)          # distance along the ray
    if lam <= MIN_LAMBDA:            # triangle is behind the seat (or at the seat itself)
        return False, lam, u, v
    return True, lam, u, v


def is_shaded(p, s, triangles):
    """True if the ray from p towards the sun hits ANY triangle."""
    return any(ray_hits_triangle(p, s, *tri)[0] for tri in triangles)


# ---------------------------------------------------------------------------
# Vectorised version
# ---------------------------------------------------------------------------
def mt_all(orig, dirs, tris):
    """
    orig (P,3), dirs (T,3), tris (F,3,3) -> λ (T,P,F), inf where the ray misses.
    Möller & Trumbore (1997): solve p + λs = V0 + u·E1 + v·E2 with Cramer's rule.
    """
    V0, E1, E2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(dirs[:, None, :], E2[None, :, :])                 # (T,F,3)
    a = np.einsum("fk,tfk->tf", E1, h)                             # (T,F)  determinant
    ok = np.abs(a) > EPS                                           # drop parallel rays
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = orig[:, None, :] - V0[None, :, :]                          # (P,F,3)
    u = f[:, None, :] * np.einsum("pfk,tfk->tpf", s, h)            # (T,P,F)
    q = np.cross(s, E1[None, :, :])                                # (P,F,3)
    vv = f[:, None, :] * np.einsum("tk,pfk->tpf", dirs, q)
    lam = f[:, None, :] * np.einsum("fk,pfk->pf", E2, q)[None]
    hit = ok[:, None, :] & (u >= 0) & (vv >= 0) & (u + vv <= 1) & (lam > MIN_LAMBDA)
    return np.where(hit, lam, np.inf)


def first_hit_t(orig, dirs, tris):
    """Distance to the nearest triangle along each ray, shape (T,P); inf = no hit."""
    return mt_all(orig, dirs, tris).min(axis=2)
