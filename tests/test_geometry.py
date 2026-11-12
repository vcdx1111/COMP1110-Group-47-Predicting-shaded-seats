"""Geometry: ear clipping, sample grids, coordinate conversion, automatic section pickup."""
import json

import numpy as np
import pytest

from shade.geometry import Feature, build_geometry, ear_clip, grid_fracs, grid_points, shoelace
from shade.model import build_scene


def tri_area(t):
    (x1, y1), (x2, y2), (x3, y3) = t
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


# --- ear clipping -------------------------------------------------------------------
L_SHAPE = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]                       # area 3
U_SHAPE = [(0, 0), (3, 0), (3, 3), (2, 3), (2, 1), (1, 1), (1, 3), (0, 3)]       # area 7


@pytest.mark.parametrize("ring, area", [(L_SHAPE, 3.0), (U_SHAPE, 7.0), (L_SHAPE[::-1], 3.0)],
                         ids=["L", "U", "L clockwise"])
def test_ear_clip_area_of_concave_polygons(ring, area):
    tris = ear_clip(ring)
    assert len(tris) == len(ring) - 2
    assert sum(tri_area(t) for t in tris) == pytest.approx(area)


def test_ear_clip_area_of_every_roof(cfg, geom):
    """The triangles of each roof must add up to the roof polygon's area."""
    for name in cfg.chain_order:
        ring = geom["polygons"][name]
        tris = ear_clip(ring)
        assert len(tris) == len(ring) - 2
        assert sum(tri_area(t) for t in tris) == pytest.approx(shoelace(ring)[0], rel=1e-9)


# --- sample grid --------------------------------------------------------------------
def test_grid_fractions():
    assert grid_fracs(3) == pytest.approx((1 / 6, 1 / 2, 5 / 6))
    assert grid_fracs(5) == pytest.approx((0.1, 0.3, 0.5, 0.7, 0.9))


@pytest.mark.parametrize("n", [3, 5])
def test_grid_points_are_cell_centres(n):
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    pts = np.array(grid_points(square, n))
    assert pts.shape == (n * n, 2)
    assert pts.mean(axis=0) == pytest.approx((0.5, 0.5))
    assert pts.min() == pytest.approx(1 / (2 * n))
    assert pts.max() == pytest.approx(1 - 1 / (2 * n))


# --- coordinates --------------------------------------------------------------------
def test_geometry_matches_reference(geom, reference_dir):
    """The digitised file must give the geometry that the reference results were made with."""
    ref = json.loads((reference_dir / "geometry_local.json").read_text(encoding="utf-8"))
    for key in ["origin_home_plate", "orientation_deg", "points", "polygons", "sample_points_xy"]:
        assert geom[key] == ref[key], key


def test_scale_checks(res):
    checks = {name: (d, ok) for name, d, ref, err, ok in res["checks"]}
    for name in ["LF_POLE", "RF_POLE", "CF"]:
        assert checks[name][1], f"{name} is outside the scale tolerance"
    assert res["foul"]["ok"]
    assert res["orient"] == pytest.approx(56.0, abs=0.1)


# --- scene --------------------------------------------------------------------------
def test_scene_triangle_counts(base_scene):
    kinds = {k: int((base_scene.kind == k).sum()) for k in ["roof", "wall", "stand"]}
    assert kinds == {"roof": 37, "wall": 52, "stand": 204}


def test_sample_points_are_at_head_height_above_the_stand(base_scene, cfg):
    for sec, P in base_scene.samples.items():
        assert P.shape == (cfg.grid_n ** 2, 3)
        assert (P[:, 2] > cfg.head_height).all()
        assert (P[:, 2] < cfg.roof_height).all()


def test_new_section_polygon_is_picked_up(cfg, res, geom):
    """A polygon named SEC_* needs no code change: it appears in the geometry and the scene."""
    quad = np.array(geom["polygons"]["SEC_122"])
    small = (quad.mean(axis=0) + 0.5 * (quad - quad.mean(axis=0))).tolist()
    extra = Feature("SEC_NEW", "polygon", lonlat=[], xy=[tuple(p) for p in small])
    other = Feature("OTHER_AREA", "polygon", lonlat=[], xy=[tuple(p) for p in small])
    res2 = dict(res, feats=res["feats"] + [extra, other])
    geom2 = build_geometry(res2, cfg.grid_n)
    assert "SEC_NEW" in geom2["sample_points_xy"]
    assert "OTHER_AREA" not in geom2["sample_points_xy"]
    scene = build_scene(geom2, cfg.variants[0], cfg)
    assert list(scene.samples) == list(geom["sample_points_xy"]) + ["SEC_NEW"]
