"""Exposure, ranks, rank stability, resolution checks and the baseline rule."""
import pandas as pd
import pytest

from shade.baseline import baseline_under_roof, compare_with_model
from shade.geometry import build_geometry
from shade.metrics import (exposure, rank_agreement, rank_stability, resolution_comparison,
                           sensitivity_summary)
from shade.model import run


def toy_q():
    """Two variants, one date, three sections, two 5-minute steps."""
    rows = []
    shade = {"base": {"A": [1.0, 1.0], "B": [0.5, 0.5], "C": [0.0, 0.0]},
             "alt": {"A": [0.0, 0.0], "B": [0.5, 0.5], "C": [0.0, 0.0]}}
    for variant, secs in shade.items():
        for sec, qs in secs.items():
            for t, q in zip(["14:00", "14:05"], qs):
                rows.append(dict(variant=variant, date="D", section=sec, time=t, q=q,
                                 sun_min=(1 - q) * 5))
    return pd.DataFrame(rows)


def test_exposure_and_rank():
    E = exposure(toy_q()).set_index(["variant", "section"])
    assert E.loc[("base", "A"), "E_min"] == 0
    assert E.loc[("base", "B"), "E_min"] == 5
    assert E.loc[("base", "C"), "E_min"] == 10
    assert list(E.loc["base", "rank"]) == [1, 2, 3]
    assert list(E.loc["alt", "rank"]) == [2, 1, 2]          # ties share the lower rank


def test_rank_stability_and_agreement():
    E = exposure(toy_q())
    st = rank_stability(E).set_index("section")
    assert not st.loc["A", "stable"] and (st.loc["A", "rank_min"], st.loc["A", "rank_max"]) == (1, 2)
    assert not st.loc["B", "stable"]
    assert not st.loc["C", "stable"]
    ag = rank_agreement(E).iloc[0]
    assert (ag["shadiest_base"], ag["shadiest_variant"]) == ("A", "B")
    assert not ag["same_shadiest"] and not ag["same_order"]


def test_sensitivity_summary_columns():
    s = sensitivity_summary(exposure(toy_q())).set_index("section")
    assert s.loc["A", "E_base"] == 0 and s.loc["A", "E_low"] == s.loc["A", "E_high"] == 10


@pytest.fixture(scope="module")
def base_runs(cfg, geom, res):
    base = cfg.variants[:1]
    ref, _ = run(geom, cfg, cfg.scenarios, base, with_points=False)
    coarse, _ = run(geom, cfg, cfg.scenarios, base, step_min=cfg.check_step_min, with_points=False)
    fine, _ = run(build_geometry(res, cfg.check_grid_n), cfg, cfg.scenarios, base, with_points=False)
    return ref, coarse, fine


def test_exposure_is_within_the_window(base_runs, cfg):
    for q in base_runs:
        E = exposure(q)
        assert E["E_min"].between(0, cfg.window_min).all()
        assert len(E) == len(cfg.scenarios) * 4


def test_resolution_checks_give_similar_exposure(base_runs):
    """A coarser time step or a finer grid must not change the picture."""
    ref, coarse, fine = (exposure(q) for q in base_runs)
    for check, E in [("10-min step", coarse), ("5x5 grid", fine)]:
        cmp_ = resolution_comparison(ref, E, check)
        assert len(cmp_) == len(ref)
        assert cmp_["E_diff"].abs().max() <= 15, check


def test_baseline_puts_no_section_in_shade(base_scene, cfg, geom):
    """No sample point lies under a roof in plan view, so the baseline rule says full sun."""
    b = baseline_under_roof(base_scene, cfg.window_min)
    assert (b["q_baseline"] == 0).all()
    assert (b["E_baseline_min"] == cfg.window_min).all()
    q, _ = run(geom, cfg, cfg.scenarios, cfg.variants[:1], with_points=False)
    c = compare_with_model(exposure(q), b)
    assert (c["rank_baseline"] == 1).all()                  # the baseline cannot rank sections
    assert (c["E_diff_min"] >= 0).all()
