"""Unit tests for the ray-triangle test on one triangle lying flat at z = 0."""
import numpy as np
import pytest

from shade.raytrace import first_hit_t, is_shaded, ray_hits_triangle

TRIANGLE = ([0, 0, 0], [1, 0, 0], [0, 1, 0])
UP = [0, 0, 1]

CASES = [
    ("hit: point below the triangle, ray goes up", [0.2, 0.2, -1], UP, True),
    ("miss: point below but outside the triangle", [0.8, 0.8, -1], UP, False),
    ("behind: triangle is behind the ray", [0.2, 0.2, 1], UP, False),
    ("parallel: ray parallel to the triangle", [0.2, 0.2, 1], [1, 0, 0], False),
    ("edge: ray through a vertex counts as hit", [0, 0, -1], UP, True),
]


@pytest.mark.parametrize("name, p, s, expected", CASES, ids=[c[0].split(":")[0] for c in CASES])
def test_ray_hits_triangle(name, p, s, expected):
    assert ray_hits_triangle(p, s, *TRIANGLE)[0] == expected


@pytest.mark.parametrize("name, p, s, expected", CASES, ids=[c[0].split(":")[0] for c in CASES])
def test_vectorised_gives_the_same_answer(name, p, s, expected):
    tris = np.array([TRIANGLE], dtype=float)
    lam = first_hit_t(np.array([p], dtype=float), np.array([s], dtype=float), tris)
    assert bool(np.isfinite(lam[0, 0])) == expected


def test_hit_distance_and_barycentric_coordinates():
    hit, lam, u, v = ray_hits_triangle([0.2, 0.3, -2], UP, *TRIANGLE)
    assert hit
    assert lam == pytest.approx(2.0)
    assert (u, v) == pytest.approx((0.2, 0.3))


def test_is_shaded_needs_only_one_hit():
    far = ([5, 5, 0], [6, 5, 0], [5, 6, 0])
    assert is_shaded([0.2, 0.2, -1], UP, [far, TRIANGLE])
    assert not is_shaded([0.2, 0.2, -1], UP, [far])
