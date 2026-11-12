"""Validation module: label rules, labeller agreement, confusion matrices, date splits."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shade import validation as V

CAL, TEST = "2026-10-15", "2026-09-20"


def test_template_has_the_required_columns_and_no_rows():
    template = Path(__file__).resolve().parents[1] / "data" / "labels_template.csv"
    df = pd.read_csv(template)
    assert list(df.columns) == V.COLUMNS
    assert len(df) == 0


# --- labeller agreement -------------------------------------------------------------
def test_cohen_kappa_textbook_example():
    # 50 items: both sun 20, a sun / b shade 5, a shade / b sun 10, both shade 15
    a = ["sun"] * 25 + ["shade"] * 25
    b = ["sun"] * 20 + ["shade"] * 5 + ["sun"] * 10 + ["shade"] * 15
    assert V.cohen_kappa(a, b) == pytest.approx(0.4)
    assert V.cohen_kappa(a, a) == pytest.approx(1.0)


def labels_frame(rows):
    df = pd.DataFrame(rows, columns=["frame_id", "zone", "label", "labeller", "excluded_reason"])
    df["date"], df["time_kst"], df["section"] = CAL, "15:00:00", "SEC_122"
    df["excluded"] = np.where(df["label"] == "unknown", "label unknown", df["excluded_reason"])
    return df


def test_only_agreed_labels_are_used():
    labels = labels_frame([
        ("f1", "front", "sun", "L1", ""), ("f1", "front", "sun", "L2", ""),         # agreed
        ("f2", "front", "sun", "L1", ""), ("f2", "front", "mixed", "L2", ""),       # disagreed
        ("f3", "front", "shade", "L1", ""),                                         # single
        ("f4", "front", "shade", "L1", ""), ("f4", "front", "unknown", "L2", ""),   # single (unknown)
        ("f5", "front", "shade", "L1", "glare"), ("f5", "front", "shade", "L2", "glare"),  # excluded
    ])
    items, pairs = V.consolidate(labels)
    assert list(items["status"]) == ["agreed", "disagreed", "single", "single", "excluded"]
    assert list(items["used"]) == [True, False, False, False, False]
    assert items.loc[0, "label"] == "sun"
    assert len(pairs) == 2
    items2, _ = V.consolidate(labels, use_single=True)
    assert list(items2["used"]) == [True, False, True, True, False]


# --- scores -------------------------------------------------------------------------
def test_scores_three_class_and_binary():
    obs = np.array(["sun", "sun", "sun", "shade", "shade", "mixed", "mixed", "sun"])
    pred = np.array(["sun", "sun", "shade", "shade", "shade", "mixed", "sun", "mixed"])
    s = V.scores(obs, pred)
    assert s["n_3class"] == 8
    assert s["accuracy_3class"] == pytest.approx(5 / 8)
    # recalls: sun 2/4, mixed 1/2, shade 2/2
    assert s["balanced_accuracy_3class"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    # binary: drop the 2 mixed labels and the 1 sun label predicted mixed -> 5 items
    assert (s["n_binary"], s["dropped_label_mixed"], s["dropped_prediction_mixed"]) == (5, 2, 1)
    assert s["accuracy_binary"] == pytest.approx(4 / 5)
    assert s["balanced_accuracy_binary"] == pytest.approx((2 / 3 + 1.0) / 2)
    # 3 items predicted shade, 1 of them was actually sun
    assert s["n_predicted_shade"] == 3
    assert s["false_shade_share"] == pytest.approx(1 / 3)


def test_confusion_matrix_layout():
    m = V.confusion(["sun", "shade", "shade"], ["sun", "sun", "shade"])
    assert list(m.index) == list(m.columns) == ["sun", "mixed", "shade"]
    assert m.loc["shade", "sun"] == 1 and m.loc["sun", "sun"] == 1 and m.values.sum() == 3


def test_classify_zone():
    assert [V.classify(k, 3) for k in range(4)] == ["sun", "mixed", "mixed", "shade"]


# --- zones --------------------------------------------------------------------------
def test_zones_are_rows_of_the_grid(base_scene):
    for sec, P in base_scene.samples.items():
        z = V.zone_points(base_scene, sec)
        assert sorted(sum(z.values(), [])) == list(range(9))
        assert all(len(idx) == 3 for idx in z.values())
        heights = [P[z[name], 2].mean() for name in ["front", "middle", "back"]]
        assert heights[0] < heights[1] < heights[2], sec
        assert all(np.ptp(P[idx, 2]) < 1.0 for idx in z.values()), sec


# --- end to end ---------------------------------------------------------------------
def make_labels(tmp_path, geom, cfg):
    """Labels copied from the model's own predictions, so the expected accuracy is 1."""
    rows = []
    for date in (CAL, TEST):
        for t in ("14:30", "15:30", "16:30"):
            for zone in V.ZONES:
                rows.append(dict(frame_id=f"{date}_{t}", section="SEC_3B_UPPER", zone=zone,
                                 date=date, time_kst=V._norm_time(t), status="agreed",
                                 label="", used=True))
    pred = V.predict(pd.DataFrame(rows), geom, cfg)
    out = []
    for _, r in pred.iterrows():
        for who in ("L1", "L2"):
            out.append(dict(frame_id=r["frame_id"], date=r["date"], source="broadcast",
                            timecode="", time_kst=r["time_kst"][:5], section=r["section"],
                            zone=r["zone"], label=r["pred_ray"], labeller=who, excluded_reason=""))
    out[1]["label"] = "mixed" if out[0]["label"] != "mixed" else "sun"      # one disagreement
    out[3]["label"] = "unknown"                                            # one unknown
    out[4]["excluded_reason"] = out[5]["excluded_reason"] = "crowd standing"
    path = tmp_path / "labels.csv"
    pd.DataFrame(out, columns=V.COLUMNS).to_csv(path, index=False)
    return path


def test_validate_end_to_end(tmp_path, geom, cfg):
    cfg2 = replace(cfg, calibration_dates=[CAL], test_dates=[TEST])
    res = V.validate(make_labels(tmp_path, geom, cfg2), geom, cfg2)

    counts = res["counts"].set_index("date")
    assert counts.loc[CAL, "rows"] == 18 and counts.loc[CAL, "rows_excluded"] == 3
    assert counts.loc[CAL, "items_disagreed"] == 1
    assert counts.loc[CAL, "items_single_labeller"] == 1
    assert counts.loc[CAL, "items_used"] == 6
    assert counts.loc[TEST, "items_used"] == 9

    sc = res["scores"]
    ray = sc[sc.model == "ray model"]
    assert set(ray["split"]) == {"calibration", "test"}
    assert (ray["accuracy_3class"] == 1.0).all()
    # calibration and test dates are never pooled: 'all' rows exist per split only
    assert sorted(zip(ray["split"], ray["date"])) == sorted(
        [("calibration", CAL), ("calibration", "all"), ("test", TEST), ("test", "all")])
    assert set(res["confusion_3class"]["observed"]) == set(V.CLASSES)

    ag = res["agreement"]
    cal = ag[(ag.split == "calibration") & (ag.date == CAL)].iloc[0]
    assert cal["n_items"] == 7 and cal["agreement_pct"] == pytest.approx(100 * 6 / 7)
    test = ag[(ag.split == "test") & (ag.date == TEST)].iloc[0]
    assert test["agreement_pct"] == 100.0


def test_date_outside_both_lists_is_an_error(tmp_path, geom, cfg):
    cfg2 = replace(cfg, calibration_dates=[CAL], test_dates=[TEST])
    path = make_labels(tmp_path, geom, cfg2)
    with pytest.raises(ValueError, match="neither"):
        V.validate(path, geom, replace(cfg, calibration_dates=[CAL], test_dates=[]))


def test_bad_label_value_is_reported(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame([dict(frame_id="f1", date=CAL, source="", timecode="", time_kst="15:00",
                       section="SEC_122", zone="front", label="sunny", labeller="L1",
                       excluded_reason="")], columns=V.COLUMNS).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sunny"):
        V.read_labels(path)
