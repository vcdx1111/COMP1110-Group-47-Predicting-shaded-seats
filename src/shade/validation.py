"""Validation of the model against labelled broadcast frames.

Labels file (CSV), one row per frame, section, zone and labeller:
    frame_id, date, source, timecode, time_kst, section, zone, label, labeller, excluded_reason

    date             YYYY-MM-DD
    time_kst         HH:MM or HH:MM:SS, Korea Standard Time
    section          a SEC_ polygon name from the stadium file
    zone             front | middle | back: one row of the section's 3 x 3 sample grid
                     (front = nearest the field)
    label            sun | shade | mixed | unknown
    labeller         a code for the person who labelled the row (use codes, not names)
    excluded_reason  empty, or why the row must not be used (glare, crowd, cloud, ...)

Rules
  * A row with an excluded_reason, or with the label "unknown", is excluded.
  * Each (frame, section, zone) is labelled by two labellers. Only labels on which both
    agree enter the confusion matrices; agreement (%) and Cohen's kappa are reported.
  * The model predicts "sun" if 0 of the zone's 3 points are shaded, "shade" if 3 of 3,
    "mixed" otherwise, at the exact time of the frame.
  * Results are given per date. Calibration and test dates are listed separately in the
    config and are never pooled together.
"""
from __future__ import annotations

from datetime import date as _date

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .baseline import under_roof
from .model import build_scene, shade_points
from .sun import sun_positions, sun_vectors

COLUMNS = ["frame_id", "date", "source", "timecode", "time_kst", "section", "zone", "label",
           "labeller", "excluded_reason"]
ZONES = ("front", "middle", "back")
CLASSES = ("sun", "mixed", "shade")
UNKNOWN = "unknown"
ITEM = ["frame_id", "section", "zone"]


# ---------------------------------------------------------------------------
# Reading and cleaning
# ---------------------------------------------------------------------------
def read_labels(path) -> pd.DataFrame:
    """Read the labels CSV, check its columns and values, and mark the excluded rows."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"labels file: missing columns {missing}")
    df = df[COLUMNS].apply(lambda col: col.str.strip())
    for c in ["zone", "label"]:
        df[c] = df[c].str.lower()

    df["excluded"] = ""
    df.loc[df["label"] == UNKNOWN, "excluded"] = "label unknown"
    has_reason = df["excluded_reason"] != ""
    df.loc[has_reason, "excluded"] = df.loc[has_reason, "excluded_reason"]

    df["time_kst"] = [_norm_time(t) or t for t in df["time_kst"]]
    used = df[df["excluded"] == ""]
    problems = []
    for i, r in df.iterrows():
        try:
            _date.fromisoformat(r["date"])
        except ValueError:
            problems.append(f"row {i + 2}: date {r['date']!r} is not YYYY-MM-DD")
    for i, r in used.iterrows():
        if r["label"] not in CLASSES:
            problems.append(f"row {i + 2}: label {r['label']!r} is not one of {CLASSES + (UNKNOWN,)}")
        if r["zone"] not in ZONES:
            problems.append(f"row {i + 2}: zone {r['zone']!r} is not one of {ZONES}")
        if _norm_time(r["time_kst"]) is None:
            problems.append(f"row {i + 2}: time_kst {r['time_kst']!r} is not HH:MM or HH:MM:SS")
        if not r["labeller"]:
            problems.append(f"row {i + 2}: no labeller")
    if problems:
        raise ValueError("labels file:\n  " + "\n  ".join(problems[:20]))
    return df


def _norm_time(text: str):
    """'14:05' or '14:05:30' -> 'HH:MM:SS'; None if the text is not a time of day."""
    parts = text.split(":")
    if len(parts) not in (2, 3) or not all(x.isdigit() for x in parts):
        return None
    h, m, sec = (int(x) for x in (parts + ["0"])[:3])
    if h > 23 or m > 59 or sec > 59:
        return None
    return f"{h:02d}:{m:02d}:{sec:02d}"


def split_of(date: str, cfg) -> str:
    """'calibration' or 'test'; a date in neither list is an error."""
    if date in cfg.calibration_dates:
        return "calibration"
    if date in cfg.test_dates:
        return "test"
    raise ValueError(f"labels file: date {date} is in neither validation.calibration_dates nor "
                     "validation.test_dates of the config")


# ---------------------------------------------------------------------------
# Two labellers: agreement and the labels that enter the matrices
# ---------------------------------------------------------------------------
def cohen_kappa(a, b, classes=CLASSES) -> float:
    """Cohen's kappa between two labellers' labels of the same items."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in classes))
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def consolidate(labels: pd.DataFrame, use_single: bool = False):
    """Combine the two labellers' rows of each (frame, section, zone).

    Returns (items, pairs):
      items  one row per item with its status: 'agreed', 'disagreed', 'single' or 'excluded',
             and used = True for the items that enter the confusion matrices
      pairs  one row per item labelled by two labellers: both labels, for the agreement table
    """
    items, pairs = [], []
    for key, g in labels.groupby(ITEM, sort=False):
        valid = g[g["excluded"] == ""]
        ref = valid if len(valid) else g
        if ref["date"].nunique() > 1 or ref["time_kst"].nunique() > 1:
            raise ValueError(f"labels file: frame {key[0]} has more than one date or time")
        head = dict(zip(ITEM, key), date=ref["date"].iat[0], time_kst=ref["time_kst"].iat[0])
        if valid["labeller"].duplicated().any():
            raise ValueError(f"labels file: {key} is labelled twice by the same labeller")
        if len(valid) > 2:
            raise ValueError(f"labels file: {key} has more than two labellers")
        if len(valid) == 2:
            v = valid.sort_values("labeller")
            la, lb = v["label"].tolist()
            pairs.append(dict(head, labeller_a=v["labeller"].iat[0], labeller_b=v["labeller"].iat[1],
                              label_a=la, label_b=lb))
            status, label = ("agreed", la) if la == lb else ("disagreed", "")
        elif len(valid) == 1:
            status, label = "single", valid["label"].iat[0]
        else:
            status, label = "excluded", ""
        used = status == "agreed" or (status == "single" and use_single)
        items.append(dict(head, status=status, label=label if used else "", used=used))
    cols_i = ITEM + ["date", "time_kst", "status", "label", "used"]
    cols_p = ITEM + ["date", "time_kst", "labeller_a", "labeller_b", "label_a", "label_b"]
    return pd.DataFrame(items, columns=cols_i), pd.DataFrame(pairs, columns=cols_p)


def agreement_table(pairs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Agreement (%) and Cohen's kappa per date, and per split over all its dates."""
    cols = ["split", "date", "labeller_a", "labeller_b", "n_items", "agreement_pct", "kappa"]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    p = pairs.assign(split=[split_of(d, cfg) for d in pairs["date"]])
    rows = []

    def add(split, date, g):
        for (a, b), h in g.groupby(["labeller_a", "labeller_b"]):
            rows.append(dict(split=split, date=date, labeller_a=a, labeller_b=b, n_items=len(h),
                             agreement_pct=100.0 * float((h["label_a"] == h["label_b"]).mean()),
                             kappa=cohen_kappa(h["label_a"], h["label_b"])))

    for (split, date), g in p.groupby(["split", "date"]):
        add(split, date, g)
    for split, g in p.groupby("split"):
        add(split, "all", g)
    return pd.DataFrame(rows, columns=cols)


def counts_table(labels: pd.DataFrame, items: pd.DataFrame, cfg) -> pd.DataFrame:
    """How many rows and items each date has, and how many survive each rule."""
    rows = []
    for date, g in items.groupby("date"):
        lab = labels[labels["date"] == date]
        rows.append(dict(split=split_of(date, cfg), date=date, rows=len(lab),
                         rows_excluded=int((lab["excluded"] != "").sum()),
                         items=len(g),
                         items_two_labellers=int(g["status"].isin(["agreed", "disagreed"]).sum()),
                         items_agreed=int((g["status"] == "agreed").sum()),
                         items_disagreed=int((g["status"] == "disagreed").sum()),
                         items_single_labeller=int((g["status"] == "single").sum()),
                         items_used=int(g["used"].sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model predictions for the labelled items
# ---------------------------------------------------------------------------
def zone_points(scene, section: str) -> dict:
    """Sample-point indices of the front, middle and back row of a section's 3 x 3 grid.

    The grid rows that run along the stand are the ones whose points share a height.
    front = lowest row (nearest the field), back = highest row.
    """
    P = scene.samples[section]
    n = int(round(len(P) ** 0.5))
    if n != 3 or n * n != len(P):
        raise ValueError("validation zones need the 3 x 3 sample grid (sampling.grid_n = 3)")
    by_t = [[k * n + s for s in range(n)] for k in range(n)]
    by_s = [[t * n + k for t in range(n)] for k in range(n)]
    height = P[:, 2]
    if np.ptp(height) < 1e-6:                 # flat section: use distance from the roof edge
        chain = LineString(scene.chain)
        height = -np.array([chain.distance(Point(p[:2])) for p in P])

    def spread(groups):
        return sum(np.ptp(height[g]) for g in groups)

    rows = by_t if spread(by_t) <= spread(by_s) else by_s
    rows = sorted(rows, key=lambda g: height[g].mean())
    return dict(zip(ZONES, rows))


def classify(n_shaded: int, n_points: int) -> str:
    if n_shaded == 0:
        return "sun"
    return "shade" if n_shaded == n_points else "mixed"


def predict(items: pd.DataFrame, geom: dict, cfg) -> pd.DataFrame:
    """Add the ray model's and the baseline's prediction to every used item."""
    scene = build_scene(geom, cfg.variants[0], cfg)
    out = items[items["used"]].copy()
    unknown = sorted(set(out["section"]) - set(scene.samples))
    if unknown:
        raise ValueError(f"labels file: sections not in the stadium file: {unknown}")
    stamps = pd.to_datetime(out["date"] + " " + out["time_kst"]).dt.tz_localize(cfg.site["tz"])
    times = pd.DatetimeIndex(sorted(stamps.unique()))
    sun = sun_positions(times, cfg.site)
    dirs = sun_vectors(sun)
    row_of = {t: i for i, t in enumerate(times)}
    shaded = {sec: shade_points(scene, sec, dirs) for sec in set(out["section"])}
    roofed = {sec: under_roof(scene, sec) for sec in set(out["section"])}
    zones = {sec: zone_points(scene, sec) for sec in set(out["section"])}

    n_ray, p_ray, p_base, elev, az = [], [], [], [], []
    for (_, r), t in zip(out.iterrows(), stamps):
        idx = zones[r["section"]][r["zone"]]
        k = int(shaded[r["section"]][row_of[t], idx].sum())
        n_ray.append(k)
        p_ray.append(classify(k, len(idx)))
        p_base.append(classify(int(roofed[r["section"]][idx].sum()), len(idx)))
        elev.append(sun["elev"].iat[row_of[t]])
        az.append(sun["az"].iat[row_of[t]])
    out["split"] = [split_of(d, cfg) for d in out["date"]]
    out["sun_elev"], out["sun_az"] = elev, az
    out["points_shaded"] = n_ray
    out["pred_ray"] = p_ray
    out["pred_baseline"] = p_base
    return out


# ---------------------------------------------------------------------------
# Confusion matrices and scores
# ---------------------------------------------------------------------------
def confusion(obs, pred, classes=CLASSES) -> pd.DataFrame:
    """Confusion matrix: rows = observed label, columns = predicted label."""
    m = pd.DataFrame(0, index=pd.Index(classes, name="observed"),
                     columns=pd.Index(classes, name="predicted"))
    for o, p in zip(obs, pred):
        m.loc[o, p] += 1
    return m


def _balanced_accuracy(m: pd.DataFrame) -> float:
    """Mean recall over the observed classes that occur."""
    totals = m.sum(axis=1)
    recalls = [m.loc[c, c] / totals[c] for c in m.index if totals[c] > 0]
    return float(np.mean(recalls)) if recalls else float("nan")


def scores(obs, pred) -> dict:
    """Scores for one date and one model.

    3-class: all used items.
    Binary: only items where the label and the prediction are both sun or shade
    (mixed rows are dropped, and counted).
    false_shade_share: of the items predicted shade, the share that was actually sun.
    """
    obs, pred = np.asarray(obs), np.asarray(pred)
    m3 = confusion(obs, pred)
    n3 = int(m3.values.sum())
    keep = (obs != "mixed") & (pred != "mixed")
    m2 = confusion(obs[keep], pred[keep], classes=("sun", "shade"))
    n2 = int(m2.values.sum())
    pred_shade = int(m2["shade"].sum())
    return dict(
        n_3class=n3,
        accuracy_3class=float(np.trace(m3.values) / n3) if n3 else float("nan"),
        balanced_accuracy_3class=_balanced_accuracy(m3),
        n_binary=n2,
        dropped_label_mixed=int((obs == "mixed").sum()),
        dropped_prediction_mixed=int(((obs != "mixed") & (pred == "mixed")).sum()),
        accuracy_binary=float(np.trace(m2.values) / n2) if n2 else float("nan"),
        balanced_accuracy_binary=_balanced_accuracy(m2),
        n_predicted_shade=pred_shade,
        false_shade_share=float(m2.loc["sun", "shade"] / pred_shade) if pred_shade else float("nan"),
    )


def _long(m: pd.DataFrame, **head) -> list:
    return [dict(head, observed=o, predicted=p, n=int(m.loc[o, p])) for o in m.index for p in m.columns]


def score_tables(pred: pd.DataFrame):
    """(scores, 3-class confusion, binary confusion) per date and per split, for both models."""
    s_rows, c3, c2 = [], [], []

    def add(split, date, g):
        for model, col in [("ray model", "pred_ray"), ("baseline", "pred_baseline")]:
            obs, p = g["label"].values, g[col].values
            head = dict(split=split, date=date, model=model)
            s_rows.append(dict(head, **scores(obs, p)))
            c3.extend(_long(confusion(obs, p), **head))
            keep = (obs != "mixed") & (p != "mixed")
            c2.extend(_long(confusion(obs[keep], p[keep], classes=("sun", "shade")), **head))

    for (split, date), g in pred.groupby(["split", "date"]):
        add(split, date, g)
    for split, g in pred.groupby("split"):
        add(split, "all", g)
    return pd.DataFrame(s_rows), pd.DataFrame(c3), pd.DataFrame(c2)


def validate(labels_path, geom: dict, cfg) -> dict:
    """Run the whole validation. Returns a dict of tables."""
    labels = read_labels(labels_path)
    for d in sorted(labels["date"].unique()):
        split_of(d, cfg)                                   # every date must belong to one split
    items, pairs = consolidate(labels, cfg.use_single_labelled)
    pred = predict(items, geom, cfg)
    sc, c3, c2 = score_tables(pred)
    return {"counts": counts_table(labels, items, cfg),
            "agreement": agreement_table(pairs, cfg),
            "scores": sc, "confusion_3class": c3, "confusion_binary": c2,
            "items": pred}
