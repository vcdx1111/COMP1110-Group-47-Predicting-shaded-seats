"""Regression: with the original settings the model must reproduce the reference results.

Original settings: 15 Apr, 15 Jul and 15 Oct 2026, 14:00-17:00 in 5-minute steps, 3 x 3 grid,
all seven variants (the regression_dates block of data/config.yaml).
"""
import numpy as np
import pandas as pd

from shade.metrics import KEYS, exposure
from shade.model import run
from shade.raytrace import first_hit_t, is_shaded
from shade.sun import sun_table, sun_vectors

TOLERANCE_MIN = 0.5


def test_exposure_matches_reference(cfg, geom, reference_dir):
    ref = pd.read_csv(reference_dir / "exposure.csv", encoding="utf-8")
    q, _ = run(geom, cfg, cfg.regression_dates, cfg.variants, with_points=False)
    E = exposure(q)
    m = E.merge(ref, on=KEYS, how="outer", suffixes=("", "_ref"), indicator=True)
    assert (m["_merge"] == "both").all(), "variants, dates or sections differ from the reference"
    assert len(m) == len(ref) == 84
    worst = (m["E_min"] - m["E_min_ref"]).abs().max()
    assert worst <= TOLERANCE_MIN, f"largest difference {worst:.3f} min"
    assert (m["rank"] == m["rank_ref"]).all()


def test_scalar_and_vectorised_ray_tests_agree(cfg, base_scene):
    """The simple one-ray loop and the numpy version must give the same answer.

    Checked for every sample point on all three dates, at every third time step
    (15-minute spacing) to keep the pure-Python loop short: 3 x 12 x 36 = 1296 rays,
    each tested against all 293 triangles.
    """
    n = agree = 0
    for d in cfg.regression_dates:
        sun = sun_table(d.date, cfg.site, cfg.start, cfg.end, cfg.step_min)
        dirs = sun_vectors(sun)[::3]
        for P in base_scene.samples.values():
            fast = np.isfinite(first_hit_t(P, dirs, base_scene.tris))
            for ti in range(len(dirs)):
                for j in range(len(P)):
                    n += 1
                    agree += is_shaded(P[j], dirs[ti], base_scene.tris) == fast[ti, j]
    assert n == 3 * 12 * 4 * 9
    assert agree == n
