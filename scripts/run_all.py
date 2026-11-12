#!/usr/bin/env python3
"""Run the whole analysis and write every table used in the report.

    python scripts/run_all.py

Reads data/config.yaml and data/stadium.geojson. Writes CSV tables to outputs/tables/ and
docs/worked_example.md. Then run scripts/make_figures.py for the figures.

Steps
  1. geometry      local coordinates, scale checks, sample points
  2. main run      every scenario date × every sensitivity variant
  3. sensitivity   exposure range and rank stability across the variants
  4. resolution    reruns with a coarser time step and with a finer sample grid
  5. baseline      "shaded only if directly under the roof" against the ray model
  6. validation    against labelled frames, if the labels file exists
  7. worked example
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from shade import validation  # noqa: E402
from shade.baseline import baseline_under_roof, compare_with_model  # noqa: E402
from shade.config import DEFAULT_CONFIG, load_config  # noqa: E402
from shade.geometry import (analyse, build_geometry, checks_table, compare_orientations,  # noqa: E402
                            sections_table)
from shade.metrics import (exposure, rank_agreement, rank_stability, resolution_comparison,  # noqa: E402
                           sensitivity_summary)
from shade.model import build_scene, run  # noqa: E402
from shade.worked_example import worked_example  # noqa: E402


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8", float_format="%.4f", lineterminator="\n")
    print(f"  wrote {path.relative_to(ROOT).as_posix()}  ({len(df)} rows)")


def report_table(sens: pd.DataFrame, cfg) -> pd.DataFrame:
    """Exposure as it appears in the report: base value and the range across the variants."""
    t = sens.assign(cell=[f"{b:.0f} ({lo:.0f}-{hi:.0f})" for b, lo, hi in
                          zip(sens["E_base"], sens["E_low"], sens["E_high"])])
    wide = t.pivot(index="section", columns="date", values="cell")
    wide = wide.reindex(index=list(dict.fromkeys(sens["section"])),
                        columns=list(dict.fromkeys(sens["date"])))
    wide.insert(0, "label", [cfg.label(s) for s in wide.index])
    wide.columns.name = None
    return wide.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.yaml")
    ap.add_argument("--compare", nargs="*", default=[], metavar="FILE",
                    help="other digitisers' GeoJSON/KML files, to compare the orientation")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()

    cfg = load_config(args.config)
    tables = cfg.root / "outputs" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    base = cfg.variants[:1]

    # 1. geometry ---------------------------------------------------------------------
    print("1. Geometry")
    res = analyse(cfg.stadium_file, cfg.official_dist, cfg.scale_tol_pct, cfg.foul_tol_deg)
    geom = build_geometry(res, cfg.grid_n)
    scene = build_scene(geom, cfg.variants[0], cfg)
    checks = checks_table(res)
    save(checks, tables / "geometry_checks.csv")
    save(sections_table(res), tables / "geometry_sections.csv")
    zrows = []
    for sec, P in scene.samples.items():
        zone_of = {i: z for z, idx in validation.zone_points(scene, sec).items() for i in idx} \
            if cfg.grid_n == 3 else {}
        for j, (x, y, z) in enumerate(P):
            zrows.append(dict(section=sec, point=j + 1, zone=zone_of.get(j, ""), x_east_m=x,
                              y_north_m=y, head_height_z_m=z))
    save(pd.DataFrame(zrows), tables / "sample_points.csv")
    kinds = {k: int((scene.kind == k).sum()) for k in ["roof", "wall", "stand"]}
    print(f"  orientation {res['orient']:.2f} deg, {len(scene.tris)} triangles {kinds}, "
          f"sections: {', '.join(scene.samples)}")
    failed = checks[checks["passed"].astype(str) == "False"]["check"].tolist()
    if failed:
        print(f"  scale checks outside tolerance: {'; '.join(failed)}")
    if args.compare:
        others = [analyse(Path(p)) for p in args.compare]
        cmp_ = compare_orientations([res] + others)
        save(pd.DataFrame(cmp_["orientations"], columns=["file", "orientation_deg"]),
             tables / "orientation_comparison.csv")
        print(f"  orientation spread {cmp_['spread_deg']:.2f} deg, suggested test range "
              f"+-{cmp_['suggested_range_deg']:.1f} deg")

    # 2. main run ---------------------------------------------------------------------
    print("2. Scenarios x variants")
    q, pts = run(geom, cfg, cfg.scenarios, cfg.variants)
    E = exposure(q)
    save(q, tables / "q_t.csv")
    save(E, tables / "exposure.csv")
    save(pts[pts.variant == "base"], tables / "points_shaded_base.csv")

    # 3. sensitivity ------------------------------------------------------------------
    print("3. Sensitivity and rank stability")
    sens = sensitivity_summary(E)
    stab = rank_stability(E)
    save(sens, tables / "sensitivity_summary.csv")
    save(stab, tables / "rank_stability.csv")
    save(rank_agreement(E), tables / "rank_agreement.csv")
    save(report_table(sens, cfg), tables / "report_exposure_table.csv")

    # 4. resolution -------------------------------------------------------------------
    print("4. Resolution checks")
    q_step, _ = run(geom, cfg, cfg.scenarios, base, step_min=cfg.check_step_min, with_points=False)
    geom_fine = build_geometry(res, cfg.check_grid_n)
    q_grid, _ = run(geom_fine, cfg, cfg.scenarios, base, with_points=False)
    reso = pd.concat([
        resolution_comparison(E, exposure(q_step), f"{cfg.check_step_min}-min steps"),
        resolution_comparison(E, exposure(q_grid), f"{cfg.check_grid_n}x{cfg.check_grid_n} grid"),
    ], ignore_index=True)
    save(reso, tables / "resolution_check.csv")
    q_alt = pd.concat([q_step.assign(check=f"{cfg.check_step_min}-min steps"),
                       q_grid.assign(check=f"{cfg.check_grid_n}x{cfg.check_grid_n} grid")])
    save(q_alt, tables / "q_t_resolution.csv")

    # 5. baseline ---------------------------------------------------------------------
    print("5. Baseline rule")
    comp = compare_with_model(E, baseline_under_roof(scene, cfg.window_min))
    save(comp, tables / "baseline_vs_model.csv")

    # 6. validation -------------------------------------------------------------------
    print("6. Validation")
    if cfg.labels_file.exists():
        val = validation.validate(cfg.labels_file, geom, cfg)
        for name, df in val.items():
            save(df, tables / f"validation_{name}.csv")
    else:
        print(f"  skipped: no labels file at {cfg.labels_file.relative_to(cfg.root).as_posix()} "
              "(copy data/labels_template.csv and fill it in)")

    # 7. worked example ---------------------------------------------------------------
    print("7. Worked example")
    doc = cfg.root / "docs" / "worked_example.md"
    doc.write_text(worked_example(geom, cfg), encoding="utf-8", newline="\n")
    print(f"  wrote {doc.relative_to(ROOT).as_posix()}")

    # summary -------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\nDirect-sun exposure, minutes out of {cfg.window_min}: base (range across the "
          f"{len(cfg.variants) - 1} sensitivity variants)")
    print(report_table(sens, cfg).drop(columns="section").to_string(index=False))
    print("\nRank stability (rank 1 = shadiest)")
    print(stab.assign(section=[cfg.label(s) for s in stab["section"]])
          .drop(columns=["base_E_min"]).to_string(index=False))
    worst = reso.loc[reso.groupby("check")["E_diff"].apply(lambda s: s.abs().idxmax())]
    print("\nResolution checks: largest change in exposure")
    for _, r in worst.iterrows():
        changed = int((~reso[reso.check == r["check"]]["same_rank"]).sum())
        print(f"  {r['check']}: {r['E_diff']:+.1f} min ({cfg.label(r['section'])}, {r['date']}); "
              f"ranks changed in {changed} of {int((reso.check == r['check']).sum())} cases")
    print(f"\nBaseline rule: {int((comp['q_baseline'] == 0).sum())} of {len(comp)} section-dates fully "
          f"sunny; largest gap to the ray model {comp['E_diff_min'].max():.0f} min")
    print(f"\nDone in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
