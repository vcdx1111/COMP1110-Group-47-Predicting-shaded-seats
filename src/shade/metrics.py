"""Summaries of the model output: exposure, ranks, sensitivity, rank stability, resolution.

All functions take the tables produced by model.run and return pandas DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["variant", "date", "section"]


def exposure(q: pd.DataFrame) -> pd.DataFrame:
    """Direct-sun exposure E_i = Σ (1 − q_i(t)) · Δt in minutes, and its rank.

    Rank 1 is the shadiest section (lowest E) of that variant and date; ties share the
    lower rank.
    """
    E = (q.groupby(KEYS, sort=False)["sun_min"].sum()
         .rename("E_min").reset_index())
    E["rank"] = E.groupby(["variant", "date"])["E_min"].rank(method="min").astype(int)
    return E


def _variant_names(E: pd.DataFrame) -> list:
    return list(dict.fromkeys(E["variant"]))


def sensitivity_summary(E: pd.DataFrame, base: str = "base") -> pd.DataFrame:
    """One row per date and section: E in every variant, and the range across the
    non-base variants (E_low, E_high)."""
    names = _variant_names(E)
    wide = (E.pivot(index=["date", "section"], columns="variant", values="E_min")
            .reindex(columns=names))
    others = [n for n in names if n != base]
    out = wide.copy()
    out["E_low"] = wide[others].min(axis=1)
    out["E_high"] = wide[others].max(axis=1)
    out["E_spread"] = out["E_high"] - out["E_low"]
    order = E[E.variant == base][["date", "section"]]
    out = out.reindex(pd.MultiIndex.from_frame(order)).reset_index()
    out.columns.name = None
    return out.rename(columns={base: "E_base"})


def rank_stability(E: pd.DataFrame, base: str = "base") -> pd.DataFrame:
    """One row per date and section: does the section keep its base-case rank?

    rank_min / rank_max   best and worst rank across all variants
    n_same                number of sensitivity variants with the same rank as the base case
    stable                True if every variant gives the base-case rank
    """
    rows = []
    for (d, sec), g in E.groupby(["date", "section"], sort=False):
        r_base = int(g.loc[g.variant == base, "rank"].iat[0])
        others = g[g.variant != base]
        n_same = int((others["rank"] == r_base).sum())
        rows.append(dict(date=d, section=sec, base_rank=r_base,
                         base_E_min=float(g.loc[g.variant == base, "E_min"].iat[0]),
                         rank_min=int(g["rank"].min()), rank_max=int(g["rank"].max()),
                         n_same=n_same, n_variants=len(others), stable=n_same == len(others)))
    return pd.DataFrame(rows)


def _spearman(a, b) -> float:
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")                          # all sections tied: no order to compare
    return float(np.corrcoef(ra, rb)[0, 1])


def rank_agreement(E: pd.DataFrame, base: str = "base") -> pd.DataFrame:
    """One row per date and sensitivity variant: is the ranking the same as the base case?"""
    rows = []
    for d, g in E.groupby("date", sort=False):
        b = g[g.variant == base].set_index("section")
        top_b = "+".join(b.index[b["rank"] == 1])
        for name in _variant_names(g):
            if name == base:
                continue
            v = g[g.variant == name].set_index("section").reindex(b.index)
            top_v = "+".join(v.index[v["rank"] == 1])
            rows.append(dict(date=d, variant=name, shadiest_base=top_b, shadiest_variant=top_v,
                             same_shadiest=top_b == top_v,
                             same_order=bool((v["rank"] == b["rank"]).all()),
                             spearman=_spearman(b["E_min"].values, v["E_min"].values)))
    return pd.DataFrame(rows)


def resolution_comparison(E_ref: pd.DataFrame, E_check: pd.DataFrame, check: str,
                          base: str = "base") -> pd.DataFrame:
    """Base-case exposure at the standard resolution against a rerun at another resolution."""
    a = E_ref[E_ref.variant == base][["date", "section", "E_min", "rank"]]
    b = E_check[E_check.variant == base][["date", "section", "E_min", "rank"]]
    m = a.merge(b, on=["date", "section"], suffixes=("_ref", "_check"))
    m.insert(0, "check", check)
    m["E_diff"] = m["E_min_check"] - m["E_min_ref"]
    m["same_rank"] = m["rank_ref"] == m["rank_check"]
    return m
