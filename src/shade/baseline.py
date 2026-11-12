"""The baseline rule from the project plan: "shaded only if directly under the roof".

A sample point counts as shaded when it lies inside a roof polygon in plan view. The rule
ignores the sun, so it gives the same answer at every time and on every date. Comparing it
with the ray model shows what the sun angle and the roof height add.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


def under_roof(scene, section: str) -> np.ndarray:
    """Boolean (P,): is each sample point of the section under a roof in plan view?"""
    polys = [Polygon(r) for r in scene.roofs_xy.values()]
    xy = scene.samples[section][:, :2]
    return np.array([any(p.contains(Point(x, y)) for p in polys) for x, y in xy])


def baseline_under_roof(scene, window_min: float) -> pd.DataFrame:
    """Shaded share and direct-sun minutes of every section under the baseline rule."""
    rows = []
    for sec in scene.samples:
        q = float(under_roof(scene, sec).mean())
        rows.append(dict(section=sec, q_baseline=q, E_baseline_min=(1 - q) * window_min))
    return pd.DataFrame(rows)


def compare_with_model(E: pd.DataFrame, baseline: pd.DataFrame, base: str = "base") -> pd.DataFrame:
    """One row per date and section: baseline exposure and rank next to the ray model's."""
    m = (E[E.variant == base][["date", "section", "E_min", "rank"]]
         .rename(columns={"E_min": "E_model_min", "rank": "rank_model"})
         .merge(baseline, on="section"))
    m["rank_baseline"] = (m.groupby("date")["E_baseline_min"].rank(method="min").astype(int))
    m["E_diff_min"] = m["E_baseline_min"] - m["E_model_min"]
    return m[["date", "section", "E_model_min", "E_baseline_min", "E_diff_min",
              "rank_model", "rank_baseline", "q_baseline"]]
