#!/usr/bin/env python3
"""Draw every figure used in the report.

    python scripts/run_all.py        # first: writes the tables
    python scripts/make_figures.py   # then: writes outputs/figures/*.png

Figures
  fig1_shade_fraction       q(t) for every scenario date, with the sensitivity range
  fig2_shade_map            plan view: which sample points are shaded at four times
  fig3_cross_section_3B     third-base stand cross-section with sun rays
  fig4_sensitivity          exposure under each sensitivity variant
  fig5_baseline_vs_model    the under-roof baseline rule against the ray model
  fig6_plan_view            the digitised stadium with the sample points
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

from shade.config import load_config  # noqa: E402
from shade.geometry import analyse, build_geometry, classify_inner, label_point, mound_xy  # noqa: E402
from shade.model import build_scene, shade_points  # noqa: E402
from shade.sun import sun_table, sun_vectors  # noqa: E402

# Colours: surface, text and grid tones, then a fixed categorical order for the sections.
# The first four section colours were checked for colour-blind separation on this surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SUN_C, SHADE_C = "#eb6834", "#184f95"
STAND_C, ROOF_C, ROOF_EDGE = "#efeeea", "#b9b8b2", "#8d8c86"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def tmin(s: str) -> int:
    h, mm = map(int, s.split(":")[:2])
    return h * 60 + mm


def section_order(cfg, sections) -> list:
    """Sections in their fixed colour order; sections not listed in the config follow."""
    first = [s for s in cfg.figures["series_order"] if s in sections]
    return first + [s for s in sections if s not in first]


def section_colours(cfg, sections) -> dict:
    order = section_order(cfg, sections)
    if len(order) > len(PALETTE):
        raise ValueError(f"{len(order)} sections but only {len(PALETTE)} colours: "
                         "split the figure into groups of sections")
    return dict(zip(order, PALETTE))


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print(f"  wrote {(out / name).relative_to(ROOT).as_posix()}")


# --- 1. q(t) small multiples ----------------------------------------------------------
def fig_shade_fraction(cfg, q, E, colours, out):
    order = list(colours)
    t0, t1 = tmin(cfg.start), tmin(cfg.end)
    n_var = q["variant"].nunique() - 1
    fig, axes = plt.subplots(1, len(cfg.scenarios), figsize=(4 * len(cfg.scenarios), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, cfg.scenarios):
        sub = q[q.date == sc.key]
        rows = []
        for sec in order:
            s = sub[sub.section == sec]
            base = s[s.variant == "base"].sort_values("time")
            band = s.groupby("time")["q"].agg(["min", "max"]).reindex(base.time)
            x = np.r_[base.time.map(tmin).values, t1]             # close the last step at the window end
            lo, hi, mid = (np.r_[v, v[-1]] for v in (band["min"].values, band["max"].values, base.q.values))
            ax.fill_between(x, lo, hi, color=colours[sec], alpha=0.14, lw=0, step="post")
            ax.step(x, mid, where="post", color=colours[sec], lw=2)
            e = E[(E.variant == "base") & (E.date == sc.key) & (E.section == sec)].E_min.iat[0]
            rows.append((sec, e))
        for k, (sec, e) in enumerate(sorted(rows, key=lambda r: r[1])):
            yy = -0.30 - 0.085 * k
            ax.plot([t0 + 2, t0 + 12], [yy, yy], color=colours[sec], lw=2.5, clip_on=False)
            ax.text(t0 + 16, yy, cfg.label(sec), fontsize=7.5, color=INK2, va="center")
            ax.text(t1, yy, f"{e:.0f} min sun", fontsize=7.5, color=INK, va="center", ha="right",
                    fontweight="bold")
        ax.text(t0, -0.20, f"Direct-sun exposure E ({cfg.start}–{cfg.end}, lowest first)", fontsize=7.5,
                color=INK2, va="center")
        ax.set_title(sc.label, loc="left", fontsize=9.5)
        ax.set_xlim(t0, t1)
        ticks = list(range(t0, t1 + 1, 60))
        ax.set_xticks(ticks, [f"{t // 60:02d}:{t % 60:02d}" for t in ticks])
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set_ylabel("share of section in shade, q(t)")
    handles = [Line2D([], [], color=colours[s], lw=2, label=cfg.label(s)) for s in order]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.15,
                                 label=f"range across {n_var} sensitivity runs"))
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.01, 1.0), fontsize=8)
    bottom = 0.30 + 0.035 * max(0, len(order) - 4)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=bottom, wspace=0.12)
    save(fig, out, "fig1_shade_fraction.png")


# --- 2. plan-view shade map -----------------------------------------------------------
def fig_shade_map(cfg, geom, scene, out):
    sc = cfg.scenario(cfg.figures["map_scenario"])
    times = cfg.figures["map_times"]
    sun = sun_table(sc.date, cfg.site, cfg.start, cfg.end, cfg.step_min)
    fig, axes = plt.subplots(1, len(times), figsize=(3.25 * len(times), 4.0))
    for ax, t in zip(np.atleast_1d(axes), times):
        at = sun.index.strftime("%H:%M") == t
        row = sun.loc[at].iloc[0]
        dirs = sun_vectors(sun.loc[at])
        for tri in scene.tris[scene.kind == "stand"]:
            ax.add_patch(MPoly(tri[:, :2], closed=True, fc=STAND_C, ec="none"))
        for r in scene.roofs_xy.values():
            ax.add_patch(MPoly(np.array(r), closed=True, fc=ROOF_C, ec=ROOF_EDGE, lw=0.6))
        for sec, P in scene.samples.items():
            ax.add_patch(MPoly(np.array(geom["polygons"][sec]), closed=True, fc="none", ec=INK2, lw=0.8))
            s = shade_points(scene, sec, dirs)[0]
            xy = P[:, :2]
            ax.scatter(xy[~s, 0], xy[~s, 1], s=26, c=SUN_C, marker="o", edgecolors=SURFACE, linewidths=0.8,
                       zorder=3)
            ax.scatter(xy[s, 0], xy[s, 1], s=24, c=SHADE_C, marker="s", edgecolors=SURFACE, linewidths=0.8,
                       zorder=3)
        for name in ["LF_POLE", "RF_POLE"]:
            x, y = geom["points"][name]
            ax.plot([0, x], [0, y], color=GRID, lw=0.8, zorder=1)
        az = math.radians(row.az)
        ax.annotate("", xy=(55 + 22 * math.sin(az), 85 + 22 * math.cos(az)), xytext=(55, 85),
                    arrowprops=dict(arrowstyle="-|>", color=SUN_C, lw=1.6))
        ax.text(50, 92, f"sun: az {row.az:.0f}°, elev {row.elev:.0f}°", ha="center", fontsize=7.5, color=INK2)
        ax.set_title(f"{sc.label} {t}", loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(-85, 100)
        ax.set_ylim(-70, 120)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax0 = np.atleast_1d(axes)[0]
    ax0.text(-20, -62, "home stand", fontsize=7, color=INK2, ha="center")
    ax0.text(-78, 60, "3B stand", fontsize=7, color=INK2, rotation=75)
    ax0.text(60, -60, "1B stand", fontsize=7, color=INK2, ha="center")
    h = [Line2D([], [], marker="o", ls="", color=SUN_C, label="sample point in direct sun"),
         Line2D([], [], marker="s", ls="", color=SHADE_C, label="sample point in shade"),
         plt.Rectangle((0, 0), 1, 1, fc=ROOF_C, ec=ROOF_EDGE, label="roof (plan view)"),
         Line2D([], [], color=SUN_C, marker=">", ls="-", label="direction towards the sun")]
    fig.legend(handles=h, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.01, 1.0), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig2_shade_map.png")


# --- 3. third-base cross-section through sections 529 and 122 --------------------------
def fig_cross_section(cfg, geom, scene, colours, out, upper="SEC_3B_UPPER", lower="SEC_122",
                      profile="3B", roof="ROOF_3B"):
    if upper not in scene.samples or lower not in scene.samples:
        print("  skipped fig3: the 3B sections are not in the stadium file")
        return
    sc = cfg.scenario(cfg.figures["map_scenario"])
    sun = sun_table(sc.date, cfg.site, cfg.start, cfg.end, cfg.step_min)
    prof = cfg.profiles[profile]
    chain = LineString(scene.chain)
    centre = np.array(geom["polygons"][upper]).mean(0)
    foot = chain.interpolate(chain.project(Point(centre)))
    n_out = np.array([foot.x, foot.y]) - centre
    n_out /= np.linalg.norm(n_out)                       # pointing from the seats towards the roof
    normal_bearing = math.degrees(math.atan2(n_out[0], n_out[1])) % 360

    def depth_column(sec):
        """The three sample points that run from the back row to the front row."""
        P = scene.samples[sec]
        d = np.array([chain.distance(Point(p[:2])) for p in P])
        groups = [[0, 3, 6], [1, 4, 7], [2, 5, 8]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        spread = [np.ptp(d[g[1]]) for g in groups]
        col = sorted(groups[int(np.argmax(spread))][1], key=lambda i: d[i])
        return [(d[i], P[i][2]) for i in col]

    ring = [tuple(p) for p in geom["polygons"][roof]]
    flags = classify_inner({k: [tuple(p) for p in geom["polygons"][k]] for k in cfg.chain_order},
                           mound_xy(geom["orientation_deg"], cfg.mound_distance))[roof]
    inner = [p for p, f in zip(ring, flags) if f]
    roof_w = Polygon(ring).area / LineString(inner).length

    fig, ax = plt.subplots(figsize=(11, 4.9))
    H = cfg.roof_height
    u = [p[0] for p in prof]
    z = [p[1] for p in prof]
    ax.fill_between(u, z, 0, color=STAND_C)
    ax.plot(u, z, color=INK2, lw=1.4)
    ax.fill_between([-roof_w, 0], [0, 0], [H, H], color="#e6e5e0")
    ax.plot([-roof_w, 0], [H, H], color="#6f6e69", lw=3)
    ax.plot([-roof_w, -roof_w], [0, H], color=ROOF_EDGE, lw=1.5)
    ax.text(-roof_w - 0.8, H, f"roof (front edge {H} m)", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(-roof_w / 2, H / 2, "building\n(under roof)", ha="center", fontsize=7.5, color=INK2)
    for txt, k, off in [("529 top row", 0, (-62, 2)), ("529 row 1", 1, (6, 4)), ("concourse", 3, (-48, 5)),
                        ("field wall", 5, (6, 0))]:
        ax.annotate(txt, (u[k], z[k]), xytext=off, textcoords="offset points", fontsize=7.5, color=INK2)
    ax.text(30, 3.4, "122 (lower deck)", fontsize=7.5, color=INK2, rotation=-17, ha="center", va="center")
    for sec in [upper, lower]:
        for d_, z_ in depth_column(sec):
            ax.plot(d_, z_, "o", color=colours[sec], ms=7, mec=SURFACE, zorder=4)
    d0, z0 = depth_column(upper)[-1]                      # front-row point of the upper deck
    for t, ls in zip(cfg.figures["cross_section_times"], ["-", "--", ":", "-."]):
        row = sun.loc[sun.index.strftime("%H:%M") == t].iloc[0]
        dA = math.radians(row.az - normal_bearing)
        if math.cos(dA) <= 0 or row.elev <= 0:
            continue                                      # the sun is on the field side: no ray to draw
        ep = math.atan(math.tan(math.radians(row.elev)) / math.cos(dA))
        run_ = min(30.0, (31.0 - z0) / math.tan(ep))
        ax.plot([d0, d0 - run_], [z0, z0 + run_ * math.tan(ep)], ls=ls, color=SUN_C, lw=1.4, zorder=3)
        ax.text(d0 - run_ - 0.6, z0 + run_ * math.tan(ep), f"{t} ({math.degrees(ep):.0f}°)",
                fontsize=7.5, color=INK2, va="center", ha="right")
    ax.set_xlim(-roof_w - 16, 56)
    ax.set_ylim(0, 32)
    ax.set_aspect("equal")
    ax.set_xlabel("distance from the roof front edge towards the field (m)")
    ax.set_ylabel("height above field (m)")
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("3B stand cross-section through 529 and 122 (base heights)", loc="left")
    h = [Line2D([], [], marker="o", ls="", color=colours[upper], label="529 sample points (head height)"),
         Line2D([], [], marker="o", ls="", color=colours[lower], label="122 sample points (head height)"),
         Line2D([], [], color=SUN_C, label=f"ray towards the sun, {sc.label} (projected)")]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=8)
    fig.text(0.01, 0.015, "Rays start at the 529 front-row point and are projected into this section "
             f"(seats → roof = {normal_bearing:.0f}°).\nThe model tests rays in 3D: the roof strip also runs "
             "along the stand, so a ray that clears the edge here can still hit the roof further along.",
             fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.19)
    save(fig, out, "fig3_cross_section_3B.png")


# --- 4. sensitivity: exposure under each variant ---------------------------------------
def factor_pairs(cfg) -> list:
    """Group the variants into factors: [(label, minus variant, plus variant), ...]."""
    base = cfg.variants[0]
    found = {}
    for v in cfg.variants[1:]:
        if v.vscale != base.vscale:
            key, label, plus = "vscale", f"vertical scale ±{abs(v.vscale - 1) * 100:.0f} %", v.vscale > 1
        elif v.overhang != base.overhang:
            key, label, plus = "overhang", f"roof overhang ±{abs(v.overhang) * 100:.0f} %", v.overhang > 0
        else:
            key, label, plus = "orient", f"orientation ±{abs(v.orient):g}°", v.orient > 0
        found.setdefault(key, {"label": label})["plus" if plus else "minus"] = v.name
    return [(f["label"], f.get("minus"), f.get("plus")) for f in found.values()]


def fig_sensitivity(cfg, E, colours, out):
    order = list(colours)
    factors = factor_pairs(cfg)
    nf = len(factors)
    fig, axes = plt.subplots(1, len(cfg.scenarios), figsize=(4 * len(cfg.scenarios), 1.0 + 1.05 * len(order)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    yticks, ylabels = [], []
    for ax_i, (ax, sc) in enumerate(zip(axes, cfg.scenarios)):
        e = E[E.date == sc.key].set_index(["variant", "section"])["E_min"]
        for i, sec in enumerate(order):
            top = i * (nf + 1.6)
            base = e[("base", sec)]
            ax.plot([base, base], [top - 0.45, top + nf - 0.55], color=INK2, lw=1.0, zorder=2)
            ax.text(base, top - 0.62, f"{base:.0f}", fontsize=7.5, color=INK, ha="center", va="bottom",
                    fontweight="bold")
            if ax_i == 0:
                ax.text(-0.03, top - 0.9, cfg.label(sec), transform=ax.get_yaxis_transform(), fontsize=8,
                        color=INK, ha="right", va="center", fontweight="bold")
            for j, (label, minus, plus) in enumerate(factors):
                y = top + j
                lo, hi = e[(minus, sec)], e[(plus, sec)]
                ax.plot([lo, hi], [y, y], color=colours[sec], lw=2.2, solid_capstyle="round", zorder=3)
                ax.plot(lo, y, "o", ms=6, mfc=SURFACE, mec=colours[sec], mew=1.6, zorder=4)
                ax.plot(hi, y, "o", ms=6, mfc=colours[sec], mec=SURFACE, mew=0.8, zorder=5)
                if ax_i == 0:
                    yticks.append(y)
                    ylabels.append(label)
        ax.set_title(sc.label, loc="left", fontsize=9.5)
        ax.set_xlim(-6, cfg.window_min + 6)
        ax.set_xticks(range(0, cfg.window_min + 1, 60))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(yticks, ylabels, fontsize=7.5)
    axes[0].set_ylim(len(order) * (nf + 1.6) - 1.4, -1.5)
    fig.supxlabel(f"direct-sun exposure E, minutes out of {cfg.window_min}", fontsize=8.5, color=INK2)
    h = [Line2D([], [], color=INK2, lw=1.0, label="base case (value shown)"),
         Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, mew=1.6, label="minus variant"),
         Line2D([], [], marker="o", ls="", mfc=INK2, mec=SURFACE, label="plus variant")]
    fig.legend(handles=h, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.12, wspace=0.08)
    save(fig, out, "fig4_sensitivity.png")


# --- 5. baseline rule against the ray model --------------------------------------------
def fig_baseline(cfg, comp, colours, out):
    order = list(colours)
    fig, axes = plt.subplots(1, len(cfg.scenarios), figsize=(4 * len(cfg.scenarios), 1.2 + 0.5 * len(order)),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, cfg.scenarios):
        c = comp[comp.date == sc.key].set_index("section")
        for i, sec in enumerate(order):
            em, eb = c.loc[sec, "E_model_min"], c.loc[sec, "E_baseline_min"]
            ax.plot([em, eb], [i, i], color=GRID, lw=2.2, zorder=1)
            ax.plot(eb, i, "s", ms=7, mfc=SURFACE, mec=INK2, mew=1.4, zorder=3)
            ax.plot(em, i, "o", ms=8, mfc=colours[sec], mec=SURFACE, mew=1.0, zorder=4)
            ax.text(em - 5, i, f"{em:.0f}", fontsize=7.5, color=INK, ha="right", va="center")
        ax.set_title(sc.label, loc="left", fontsize=9.5)
        ax.set_xlim(-22, cfg.window_min + 8)
        ax.set_xticks(range(0, cfg.window_min + 1, 60))
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(order)), [cfg.label(s) for s in order], fontsize=8)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    fig.supxlabel(f"direct-sun exposure E, minutes out of {cfg.window_min}", fontsize=8.5, color=INK2)
    h = [Line2D([], [], marker="s", ls="", mfc=SURFACE, mec=INK2, mew=1.4,
                label='baseline rule: "shaded only if directly under the roof"'),
         Line2D([], [], marker="o", ls="", mfc=INK2, mec=SURFACE, label="ray model (value shown)")]
    fig.legend(handles=h, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 1.0), fontsize=8)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.80, bottom=0.20, wspace=0.08)
    save(fig, out, "fig5_baseline_vs_model.png")


# --- 6. plan view of the digitised stadium ---------------------------------------------
def fig_plan_view(res, geom, out):
    fig, ax = plt.subplots(figsize=(8, 8))
    for f in res["feats"]:
        if f.kind == "polygon":
            is_roof = f.name.upper().startswith("ROOF")
            ax.add_patch(MPoly(np.array(f.xy), closed=True, fc=ROOF_C if is_roof else "#cde2fb",
                               ec=ROOF_EDGE if is_roof else "#2a78d6", lw=1, alpha=0.8))
            cx, cy = label_point(f.xy, strip=is_roof)
            if is_roof:
                ax.annotate(f.name, (cx, cy), ha="center", va="center", fontsize=7, color=INK)
            else:
                ax.annotate(f.name, (cx, max(p[1] for p in f.xy) + 1.5), ha="center", va="bottom",
                            fontsize=7, color=INK)
        elif f.kind == "line":
            ax.plot([p[0] for p in f.xy], [p[1] for p in f.xy], color=INK2, lw=2)
            ax.annotate(f.name, f.xy[0], fontsize=7, color=INK)
    for f in res["feats"]:
        if f.kind == "point":
            x, y = f.xy[0]
            ax.plot(x, y, "o", color=INK, ms=5)
            ax.annotate(f.name, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    for pts in geom["sample_points_xy"].values():
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", color=SUN_C, ms=5)
    if res["orient"] is not None:
        br = math.radians(res["orient"])
        ax.plot([0, 130 * math.sin(br)], [0, 130 * math.cos(br)], "--", color=INK2, lw=1)
    ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.87), xycoords="axes fraction",
                textcoords="axes fraction", arrowprops=dict(arrowstyle="->", color=INK2),
                ha="center", va="top", fontsize=11)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlabel("x east (m)")
    ax.set_ylabel("y north (m)")
    title = "Digitised stadium (plan view)"
    if res["orient"] is not None:
        title += f": home → centre field = {res['orient']:.1f}°"
    ax.set_title(title, loc="left")
    h = [plt.Rectangle((0, 0), 1, 1, fc=ROOF_C, ec=ROOF_EDGE, label="roof"),
         plt.Rectangle((0, 0), 1, 1, fc="#cde2fb", ec="#2a78d6", label="seating section"),
         Line2D([], [], marker=".", ls="", color=SUN_C, label="sample point"),
         Line2D([], [], ls="--", color=INK2, label="home → centre field")]
    ax.legend(handles=h, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig6_plan_view.png")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = load_config()
    tables = cfg.root / "outputs" / "tables"
    out = cfg.root / "outputs" / "figures"
    needed = ["q_t.csv", "exposure.csv", "baseline_vs_model.csv"]
    if not all((tables / n).exists() for n in needed):
        sys.exit("Tables not found. Run  python scripts/run_all.py  first.")
    out.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(tables / "q_t.csv")
    E = pd.read_csv(tables / "exposure.csv")
    comp = pd.read_csv(tables / "baseline_vs_model.csv")

    res = analyse(cfg.stadium_file, cfg.official_dist, cfg.scale_tol_pct, cfg.foul_tol_deg)
    geom = build_geometry(res, cfg.grid_n)
    scene = build_scene(geom, cfg.variants[0], cfg)
    colours = section_colours(cfg, list(scene.samples))

    print("Figures")
    fig_shade_fraction(cfg, q, E, colours, out)
    fig_shade_map(cfg, geom, scene, out)
    fig_cross_section(cfg, geom, scene, colours, out)
    fig_sensitivity(cfg, E, colours, out)
    fig_baseline(cfg, comp, colours, out)
    fig_plan_view(res, geom, out)


if __name__ == "__main__":
    main()
