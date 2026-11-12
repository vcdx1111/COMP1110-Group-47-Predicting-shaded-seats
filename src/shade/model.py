"""The shade model: build the stadium as opaque triangles, then test rays towards the sun.

Model summary
  1. Sun position: pvlib (NREL SPA), Asia/Seoul, start of every step in the game window.
  2. Stadium: a set of opaque triangles
       - roofs      satellite polygons, ear-clipped, flat plates at the roof height
       - outer wall vertical walls from the ground to the roof along the roofs' outer edges
       - stands     surfaces built from the cross-section profile (d, z) of each side,
                    running from the roofs' field-side edge towards the field
  3. Test: a ray p + λs (λ > 0) from each sample point p towards the sun s.
     If it hits any triangle the point is in shade (Möller-Trumbore).
  4. Summary: q_i(t) = shaded points / all points of section i,
              E_i = Σ (1 − q_i(t)) · Δt   [minutes of direct sun]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

from .geometry import (build_chain, chain_normals, classify_inner, clip_profile, ear_clip,
                       mound_xy, ray_segment_t)
from .raytrace import first_hit_t
from .sun import sun_table, sun_vectors


@dataclass(frozen=True)
class Variant:
    """One sensitivity variant of the geometry."""
    name: str
    vscale: float = 1.0      # vertical scale (stands and roof together)
    overhang: float = 0.0    # move the roof's field-side edge towards the field (× roof width)
    orient: float = 0.0      # change of stadium orientation (degrees, clockwise +)


@dataclass
class Scene:
    tris: np.ndarray                 # (F, 3, 3) every occluding triangle
    kind: np.ndarray                 # (F,) 'roof' | 'wall' | 'stand'
    roofs_xy: dict                   # roof polygons in plan, with the variant applied
    chain: np.ndarray                # field-side roof edge, first base -> home -> third base
    normals: np.ndarray              # unit normals of the chain, towards the field
    profiles: list                   # stand profile at every chain vertex
    samples: dict = field(default_factory=dict)   # section -> (P, 3) head-height points


def build_scene(geom: dict, v: Variant, cfg) -> Scene:
    """Triangles and sample points for one variant.

    geom is the output of geometry.build_geometry; cfg supplies the dimensions
    (roof height, stand profiles, head height, ...) from data/config.yaml.
    """
    chain_order = cfg.chain_order
    roofs = {k: [tuple(p) for p in geom["polygons"][k]] for k in chain_order}
    pts = geom["points"]
    orient = geom["orientation_deg"]
    mound = mound_xy(orient, cfg.mound_distance)
    flags = classify_inner(roofs, mound)
    chain, side, ring_ref = build_chain(roofs, flags, chain_order, cfg.roof_side)
    nrm = chain_normals(chain, mound)

    home = np.array([0.0, 0.0])
    foul = [(home, np.array(pts["LF_POLE"])), (home, np.array(pts["RF_POLE"]))]
    profiles = []
    for i, s in enumerate(side):
        prof = cfg.profiles[s]
        dmax = min(ray_segment_t(chain[i], nrm[i], a, b) for a, b in foul) - cfg.foul_margin
        prof = clip_profile(prof, max(dmax, 1.0))
        profiles.append([(d, z * v.vscale) for d, z in prof])

    tris, kind = [], []

    # Stand surface: join neighbouring chain vertices profile point by profile point;
    # each quad becomes two triangles
    n_prof = len(profiles[0])
    for i in range(len(chain) - 1):
        for k in range(n_prof - 1):
            (da, za), (db, zb) = profiles[i][k], profiles[i][k + 1]
            (dc, zc), (dd, zd) = profiles[i + 1][k], profiles[i + 1][k + 1]
            P0 = np.r_[chain[i] + da * nrm[i], za]
            P1 = np.r_[chain[i] + db * nrm[i], zb]
            P2 = np.r_[chain[i + 1] + dc * nrm[i + 1], zc]
            P3 = np.r_[chain[i + 1] + dd * nrm[i + 1], zd]
            tris += [(P0, P1, P3), (P0, P3, P2)]
            kind += ["stand", "stand"]

    # Overhang variant: move the inner vertices along the normal by (roof width × fraction)
    roofs_v = {k: [list(p) for p in r] for k, r in roofs.items()}
    if v.overhang:
        for name in chain_order:
            inner = [p for p, f in zip(roofs[name], flags[name]) if f]
            width = Polygon(roofs[name]).area / LineString(inner).length
            for ci, refs in enumerate(ring_ref):
                for rname, ri in refs:
                    if rname == name:
                        roofs_v[name][ri] = list(np.array(roofs[name][ri]) + v.overhang * width * nrm[ci])

    H = cfg.roof_height * v.vscale
    for name in chain_order:
        ring = roofs_v[name]
        for a, b, c in ear_clip(ring):
            tris.append((np.r_[a, H], np.r_[b, H], np.r_[c, H]))
            kind.append("roof")
        # Outer wall: every outline edge that is not on the inner edge, ground to roof
        f = flags[name]
        for j in range(len(ring)):
            jn = (j + 1) % len(ring)
            if f[j] and f[jn]:
                continue
            a, b = np.array(ring[j]), np.array(ring[jn])
            A0, A1, B0, B1 = np.r_[a, 0], np.r_[a, H], np.r_[b, 0], np.r_[b, H]
            tris += [(A0, B0, B1), (A0, B1, A1)]
            kind += ["wall", "wall"]

    scene = Scene(np.array(tris, dtype=float), np.array(kind), roofs_v, chain, nrm, profiles)

    # Sample points: plan position from the section grid, z = stand surface + head height.
    # Every section in the geometry is used, so a new SEC_ polygon needs no code change.
    stand = scene.tris[scene.kind == "stand"]
    for sec, grid in geom["sample_points_xy"].items():
        xy = np.array(grid, dtype=float)
        top = np.c_[xy, np.full(len(xy), 200.0)]
        t = first_hit_t(top, np.array([[0.0, 0.0, -1.0]]), stand)[0]
        if np.any(~np.isfinite(t)):
            raise ValueError(f"{sec}: no stand surface under {int((~np.isfinite(t)).sum())} of its "
                             "sample points (is the section drawn in front of a roof edge?)")
        scene.samples[sec] = np.c_[xy, 200.0 - t + cfg.head_height]
    return scene


def shade_points(scene: Scene, section: str, dirs: np.ndarray) -> np.ndarray:
    """Boolean (T, P): is sample point j of the section shaded for sun direction i?"""
    return np.isfinite(first_hit_t(scene.samples[section], dirs, scene.tris))


def run(geom: dict, cfg, dates, variants, step_min: int | None = None, with_points: bool = True):
    """Run the model for every variant, date and section.

    dates     scenarios with .key and .date
    variants  list of Variant
    step_min  time step in minutes (default: the value in the config)

    Returns (q, points):
      q       one row per variant, date, section and time step: sun angles, shaded share q
              and sun_min, the minutes of direct sun that the step contributes
      points  one row per sample point and time step (None if with_points is False)
    """
    step = cfg.step_min if step_min is None else step_min
    rows_q, rows_pt = [], []
    for v in variants:
        sc = build_scene(geom, v, cfg)
        for d in dates:
            sun = sun_table(d.date, cfg.site, cfg.start, cfg.end, step)
            dirs = sun_vectors(sun, v.orient)
            day = sun["elev"].values > 0
            for sec in sc.samples:
                shaded = shade_points(sc, sec, dirs)                        # (T,P)
                shaded[~day] = False                 # after sunset nothing counts as "shade"
                for ti, ts in enumerate(sun.index):
                    rows_q.append(dict(variant=v.name, date=d.key, section=sec, time=ts.strftime("%H:%M"),
                                       elev=sun["elev"].iat[ti], az=sun["az"].iat[ti], daylight=bool(day[ti]),
                                       q=shaded[ti].mean()))
                    if with_points:
                        for j in range(shaded.shape[1]):
                            rows_pt.append(dict(variant=v.name, date=d.key, section=sec, point=j + 1,
                                                time=ts.strftime("%H:%M"), shaded=bool(shaded[ti, j])))
    q = pd.DataFrame(rows_q)
    q["sun_min"] = (1 - q["q"]) * step * q["daylight"]
    return q, (pd.DataFrame(rows_pt) if with_points else None)
