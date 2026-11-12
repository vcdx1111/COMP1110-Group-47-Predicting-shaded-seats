"""The worked example must keep the numbers that were presented at the check-in."""
import numpy as np
import pytest

from shade.raytrace import mt_all
from shade.sun import sun_table, sun_vectors
from shade.worked_example import mt_steps, worked_example


def ray_at(cfg, base_scene, date, hhmm):
    p = base_scene.samples["SEC_3B_UPPER"][4]              # middle point of section 529
    sun = sun_table(date, cfg.site, cfg.start, cfg.end, cfg.step_min)
    at = sun.index.strftime("%H:%M") == hhmm
    s = sun_vectors(sun.loc[at])[0]
    return p, s, sun.loc[at].iloc[0]


def test_sample_point(base_scene):
    p = base_scene.samples["SEC_3B_UPPER"][4]
    assert p == pytest.approx([-46.87, 56.78, 18.53], abs=0.005)


def test_15_oct_1630_is_shaded_by_the_roof(cfg, base_scene):
    p, s, row = ray_at(cfg, base_scene, "2026-10-15", "16:30")
    assert (row.elev, row.az) == pytest.approx((16.44, 246.61), abs=0.005)
    lam = mt_all(p[None], s[None], base_scene.tris)[0, 0]
    k = int(np.argmin(lam))
    assert base_scene.kind[k] == "roof"
    st = mt_steps(p, s, base_scene.tris[k])
    assert st["u"] == pytest.approx(0.3724, abs=5e-5)
    assert st["v"] == pytest.approx(0.1489, abs=5e-5)
    assert st["lam"] == pytest.approx(14.041, abs=5e-4)


def test_15_oct_1500_is_sunny(cfg, base_scene):
    p, s, row = ray_at(cfg, base_scene, "2026-10-15", "15:00")
    assert (row.elev, row.az) == pytest.approx((32.02, 229.13), abs=0.005)
    assert not np.isfinite(mt_all(p[None], s[None], base_scene.tris)).any()


def test_document_has_both_dates(cfg, geom):
    text = worked_example(geom, cfg)
    for heading in ["Case A (sunny): 15 Oct 15:00 KST", "Case B (shaded): 15 Oct 16:30 KST",
                    "Case C (sunny): 20 Sep 15:00 KST", "Case D (shaded): 20 Sep 16:30 KST"]:
        assert heading in text
    assert "**0.3724**" in text and "**0.1489**" in text and "**14.041 m**" in text
    assert "E = 107 minutes" in text
