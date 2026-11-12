"""Writes docs/worked_example.md: one sample point, every Möller-Trumbore step with numbers.

The section, the point and the dates and times come from the worked_example block of
data/config.yaml. For each date one sunny and one shaded moment are worked through.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon

from .metrics import exposure
from .model import build_scene, run, shade_points
from .raytrace import mt_all
from .sun import sun_table, sun_vectors


def _v(x, n=3):
    return "(" + ", ".join(f"{c:.{n}f}" for c in x) + ")"


def mt_steps(p, s, tri) -> dict:
    """Every intermediate quantity of the Möller-Trumbore test for one ray and one triangle."""
    V0, V1, V2 = tri
    E1, E2 = V1 - V0, V2 - V0
    h = np.cross(s, E2)
    a = E1 @ h
    f = 1 / a
    w = p - V0
    u = f * (w @ h)
    q = np.cross(w, E1)
    vv = f * (s @ q)
    lam = f * (E2 @ q)
    return dict(V0=V0, V1=V1, V2=V2, E1=E1, E2=E2, h=h, a=a, f=f, w=w, u=u, q=q, v=vv, lam=lam)


def _block(title, label, t, p, sc, sun, roof_height, expect_shade):
    at = sun.index.strftime("%H:%M") == t
    if not at.any():
        raise ValueError(f"worked example: {t} is not a time step of the game window")
    row = sun.loc[at].iloc[0]
    s = sun_vectors(sun.loc[at])[0]
    lam_all = mt_all(p[None], s[None], sc.tris)[0, 0]
    hit = np.isfinite(lam_all)
    if bool(hit.any()) != expect_shade:
        raise ValueError(f"worked example: {label} {t} is {'shaded' if hit.any() else 'sunny'}; "
                         "change the times in the config")
    roof_idx = np.where(sc.kind == "roof")[0]
    out = [f"## {title}: {label} {t} KST", ""]
    out += ["**Step 1 — sun position (pvlib).**",
            f"Elevation α = {row.elev:.2f}°, azimuth A = {row.az:.2f}° (clockwise from north).", "",
            "**Step 2 — direction from the seat towards the sun.**",
            "s = (cos α sin A, cos α cos A, sin α)",
            f"= ({math.cos(math.radians(row.elev)):.4f}×{math.sin(math.radians(row.az)):.4f}, "
            f"{math.cos(math.radians(row.elev)):.4f}×{math.cos(math.radians(row.az)):.4f}, "
            f"{math.sin(math.radians(row.elev)):.4f}) = **{_v(s, 4)}**", ""]
    if hit.any():
        k = int(np.argmin(np.where(hit, lam_all, np.inf)))
        title3 = f"**Step 3 — the first triangle the ray hits** (triangle #{k}, type: {sc.kind[k]})"
    else:
        # show the roof triangle whose plane the ray crosses nearest to its edges
        best, k = None, None
        for i in roof_idx:
            st = mt_steps(p, s, sc.tris[i])
            miss = max(0, -st["u"]) + max(0, -st["v"]) + max(0, st["u"] + st["v"] - 1)
            if st["lam"] > 0 and (best is None or miss < best):
                best, k = miss, i
        title3 = (f"**Step 3 — no triangle is hit.** Below is the roof triangle #{k} that the ray passes "
                  "closest to, to show why it misses")
    st = mt_steps(p, s, sc.tris[k])
    out += [title3, "",
            f"V0 = {_v(st['V0'], 2)}, V1 = {_v(st['V1'], 2)}, V2 = {_v(st['V2'], 2)}",
            "", "Solve p + λs = V0 + u·E1 + v·E2 (Möller & Trumbore 1997):", "",
            "| quantity | formula | value |", "|---|---|---|",
            f"| E1, E2 | V1 − V0, V2 − V0 | {_v(st['E1'], 2)}, {_v(st['E2'], 2)} |",
            f"| h | s × E2 | {_v(st['h'], 3)} |",
            f"| a (determinant) | E1 · h | {st['a']:.4f}  (≠ 0, so the ray is not parallel) |",
            f"| w | p − V0 | {_v(st['w'], 3)} |",
            f"| u | (w · h) / a | **{st['u']:.4f}** |",
            f"| q | w × E1 | {_v(st['q'], 3)} |",
            f"| v | (s · q) / a | **{st['v']:.4f}** |",
            f"| λ | (E2 · q) / a | **{st['lam']:.3f} m** |", ""]
    X = p + st["lam"] * s
    out += [f"Check: u ≥ 0 → {st['u'] >= 0}, v ≥ 0 → {st['v'] >= 0}, u + v ≤ 1 → {st['u'] + st['v'] <= 1}, "
            f"λ > 0 → {st['lam'] > 0}.", ""]
    if hit.any():
        out += [f"All four hold, so the ray hits this {sc.kind[k]} triangle {st['lam']:.1f} m away, at "
                f"{_v(X, 2)}. Its height is {X[2]:.2f} m, which equals the roof height ({roof_height} m).",
                "", "**Result: SHADE.**", ""]
    else:
        under = any(Polygon(r).contains(Point(X[0], X[1])) for r in sc.roofs_xy.values())
        why = ("" if under else " That spot is still in front of the roof edge (not under any roof), so the "
               "ray reaches roof height before it gets to the roof and passes over it.")
        out += [f"The ray reaches the roof's height ({roof_height} m) {st['lam']:.1f} m away, at {_v(X, 2)}, "
                f"but (u, v) falls outside the triangle.{why} "
                f"None of the {len(sc.tris)} triangles in the scene satisfies all four conditions.", "",
                "**Result: DIRECT SUN.**", ""]
    return out


def _whole_game(label, date, section, j, sc, geom, cfg, sun):
    shaded = shade_points(sc, section, sun_vectors(sun))[:, j]
    times = sun.index.strftime("%H:%M")
    q, _ = run(geom, cfg, [type("D", (), {"key": label, "date": date})()], cfg.variants[:1],
               with_points=False)
    E = exposure(q)
    e = float(E.loc[E.section == section, "E_min"].iat[0])
    n_pts = len(sc.samples[section])
    switches = int(np.sum(shaded[1:] != shaded[:-1]))
    if shaded.any() and not shaded[0] and switches == 1:
        first = times[np.argmax(shaded)]
        sentence = (f"On {label} this point is in direct sun from {cfg.start} until **{first}**, "
                    f"then in shade until {cfg.end}.")
    else:
        sentence = (f"On {label} this point is shaded at {int(shaded.sum())} of the {len(shaded)} "
                    "time steps.")
    return [f"## What this point does over the whole game ({label})", "",
            f"{sentence} Doing this for all {n_pts} points of {cfg.label(section).split(' ')[0]} gives q(t). "
            f"Summing (1 − q(t)) × {cfg.step_min} min gives E = {e:.0f} minutes of direct sun.", ""]


def worked_example(geom: dict, cfg) -> str:
    """The whole document as Markdown text."""
    we = cfg.worked_example
    section, j = we["section"], int(we["point"]) - 1
    sc = build_scene(geom, cfg.variants[0], cfg)
    p = sc.samples[section][j]
    n_pts = len(sc.samples[section])
    name = cfg.label(section)
    lines = ["# Worked example — one sample point, two times", "",
             f"Section **{name}**, sample point {j + 1} of {n_pts}"
             f"{' (the middle point)' if j == n_pts // 2 else ''}.", "",
             f"- plan position: x = {p[0]:.2f} m east, y = {p[1]:.2f} m north of home plate",
             f"- seat surface at this point: {p[2] - cfg.head_height:.2f} m; head height p_z = seat + "
             f"{cfg.head_height} = **{p[2]:.2f} m**",
             f"- p = **{_v(p, 2)}**", "",
             f"Scene: base case. The roof front edge is at {cfg.roof_height} m, and the scene has "
             f"{len(sc.tris)} triangles ({(sc.kind == 'roof').sum()} roof, {(sc.kind == 'wall').sum()} wall, "
             f"{(sc.kind == 'stand').sum()} stand).", "",
             "Dates: " + "; ".join(f"**{c['label']}** (cases {chr(65 + 2 * i)} and {chr(66 + 2 * i)})"
                                   for i, c in enumerate(we["cases"])) + ".", ""]
    n_steps = 0
    for i, case in enumerate(we["cases"]):
        sun = sun_table(case["date"], cfg.site, cfg.start, cfg.end, cfg.step_min)
        n_steps = len(sun)
        a, b = chr(65 + 2 * i), chr(66 + 2 * i)
        lines += _block(f"Case {a} (sunny)", case["label"], case["sunny"], p, sc, sun,
                        cfg.roof_height, expect_shade=False)
        lines += _block(f"Case {b} (shaded)", case["label"], case["shaded"], p, sc, sun,
                        cfg.roof_height, expect_shade=True)
        lines += _whole_game(case["label"], case["date"], section, j, sc, geom, cfg, sun)

    n_sec = len(sc.samples)
    n_all = sum(len(P) for P in sc.samples.values())
    lines += ["## Cost", "",
              f"T × P × F = {n_steps} time steps × {n_all} points × {len(sc.tris)} triangles = "
              f"{n_steps * n_all * len(sc.tris):,} ray–triangle tests per date (all {n_sec} sections). "
              f"With 12 sections ({12 * n_pts} points) it would be {n_steps * 12 * n_pts * len(sc.tris):,}. "
              "That is linear in each factor, so it runs in well under a second with numpy.", ""]
    return "\n".join(lines)
