"""Stadium geometry: satellite coordinates -> local metres, sample grids, roof triangles.

Part 1 turns latitude/longitude digitised on a satellite map (GeoJSON or KML) into local
coordinates in metres with home plate at the origin (x = east, y = north), and checks the
result against published distances.

Part 2 holds the plane-geometry helpers the scene builder needs: the field-side edge of the
roofs, its normals, the stand profile clipping and ear clipping of the roof polygons.

Name rules for the digitised features (the "name" property of each point or polygon)
  HOME         home plate (required, becomes the origin)
  CF           centre of the centre-field fence
  2B           second base
  LF_POLE      left-field foul pole
  RF_POLE      right-field foul pole
  STAND_FRONT  front of the stands directly behind home plate
  ROOF_xxx     roof outline (polygon)
  SEC_xxx      seating section outline (polygon with 4 vertices). Every SEC_ polygon in
               the file is picked up and gets its own grid of sample points.
"""
from __future__ import annotations

import json
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

M_PER_DEG_LAT = 110_574.0          # one degree of latitude ≈ 110,574 m
M_PER_DEG_LON_EQ = 111_320.0       # one degree of longitude at the equator (× cos latitude)


@dataclass
class Feature:
    name: str
    kind: str                      # point | line | polygon
    lonlat: list                   # [(lon, lat), ...]
    example: bool = False
    xy: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1a. Reading the digitised file
# ---------------------------------------------------------------------------
def _drop_closing(ring):
    if len(ring) > 1 and ring[0] == ring[-1]:
        return ring[:-1]
    return ring


def parse_geojson(path: Path) -> list[Feature]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = data.get("features", [data]) if isinstance(data, dict) else []
    out = []
    for f in feats:
        props = f.get("properties") or {}
        name = str(props.get("name") or props.get("Name") or "").strip()
        ex = bool(props.get("example"))
        geom = f.get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "Point":
            out.append(Feature(name, "point", [tuple(coords[:2])], ex))
        elif gtype == "LineString":
            out.append(Feature(name, "line", [tuple(c[:2]) for c in coords], ex))
        elif gtype == "Polygon":
            out.append(Feature(name, "polygon",
                               _drop_closing([tuple(c[:2]) for c in coords[0]]), ex))
        elif gtype == "MultiPolygon":
            out.append(Feature(name, "polygon",
                               _drop_closing([tuple(c[:2]) for c in coords[0][0]]), ex))
        else:
            warnings.warn(f"'{name}': unsupported geometry type {gtype}, skipped")
    return out


def parse_kml(path: Path) -> list[Feature]:
    root = ET.parse(path).getroot()
    out = []

    def find_coords(pm, geom_suffix):
        for el in pm.iter():
            if el.tag.endswith(geom_suffix):
                for c in el.iter():
                    if c.tag.endswith("coordinates") and c.text:
                        return [tuple(map(float, t.split(",")[:2]))
                                for t in c.text.split()]
        return None

    for pm in root.iter():
        if not pm.tag.endswith("Placemark"):
            continue
        name = ""
        for ch in pm:
            if ch.tag.endswith("name") and ch.text:
                name = ch.text.strip()
                break
        if (c := find_coords(pm, "Polygon")):
            out.append(Feature(name, "polygon", _drop_closing(c)))
        elif (c := find_coords(pm, "LineString")):
            out.append(Feature(name, "line", c))
        elif (c := find_coords(pm, "Point")):
            out.append(Feature(name, "point", c[:1]))
    return out


def load_features(path: Path) -> list[Feature]:
    path = Path(path)
    if path.suffix.lower() == ".kml":
        return parse_kml(path)
    return parse_geojson(path)


# ---------------------------------------------------------------------------
# 1b. Coordinate helpers
# ---------------------------------------------------------------------------
def to_local(lon, lat, lon0, lat0):
    """Latitude/longitude -> (x east, y north) in metres from home plate.

    A flat-earth approximation; the error is negligible at stadium scale.
    """
    x = (lon - lon0) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat0))
    y = (lat - lat0) * M_PER_DEG_LAT
    return x, y


def bearing_deg(x, y):
    """Bearing from the origin to (x, y): 0° = true north, clockwise (same as the sun azimuth)."""
    return math.degrees(math.atan2(x, y)) % 360.0


def ang_diff(a, b):
    """Signed difference a − b between two bearings, in (−180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d if d != -180.0 else 180.0


def to_stadium_frame(x, y, b):
    """(east, north) -> (u, v): v points from home to centre field, u towards first base."""
    br = math.radians(b)
    return x * math.cos(br) - y * math.sin(br), x * math.sin(br) + y * math.cos(br)


def rotate_cw(x, y, deg):
    """Rotate clockwise about home plate: the stadium with its bearing changed by +deg."""
    r = math.radians(deg)
    return x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)


def shoelace(pts):
    """Area and centroid of a polygon (shoelace formula)."""
    a, cx, cy = 0.0, 0.0, 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    a *= 0.5
    if abs(a) < 1e-9:
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        return 0.0, (mx, my)
    return abs(a), (cx / (6 * a), cy / (6 * a))


def grid_fracs(n: int) -> tuple:
    """Cell-centre positions of an n × n grid: n = 3 gives (1/6, 1/2, 5/6)."""
    return tuple((2 * k + 1) / (2 * n) for k in range(n))


def grid_points(quad, n: int = 3):
    """n × n sample points inside a 4-vertex section (vertices in order around the outline).

    Each point is the centre of one grid cell (bilinear interpolation between the corners).
    Point number k (0-based) is row k // n, column k % n of the grid.
    """
    A, B, C, D = quad
    fr = grid_fracs(n)
    pts = []
    for t in fr:
        for s in fr:
            x = (1-s)*(1-t)*A[0] + s*(1-t)*B[0] + s*t*C[0] + (1-s)*t*D[0]
            y = (1-s)*(1-t)*A[1] + s*(1-t)*B[1] + s*t*C[1] + (1-s)*t*D[1]
            pts.append((x, y))
    return pts


def point_in_poly(x, y, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def label_point(pts, strip=False):
    """Where to put a label in a figure: the centroid, or for a roof strip the midpoint
    between two vertices on opposite sides of the strip."""
    _, (cx, cy) = shoelace(pts)
    if not strip and point_in_poly(cx, cy, pts):
        return cx, cy
    n = len(pts)
    for k in (n // 4, n // 2 - 1, 1):
        a, b = pts[k], pts[n - 1 - k]
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        if point_in_poly(mx, my, pts):
            return mx, my
    return pts[0]


def side_label(phi):
    """Stadium side from the angle to the centre-field axis."""
    if abs(phi) >= 157.5:
        return "behind home"
    return "1B side" if phi > 0 else "3B side"


# ---------------------------------------------------------------------------
# 1c. Analysis of one digitised file
# ---------------------------------------------------------------------------
def analyse(path: Path, official_dist: dict | None = None, scale_tol_pct: float = 2.0,
            foul_tol_deg: float = 1.5) -> dict:
    """Read a digitised file, convert to local metres and run the scale checks."""
    path = Path(path)
    feats = load_features(path)
    by_name = {f.name.upper(): f for f in feats}
    if "HOME" not in by_name:
        raise ValueError(f"{path.name}: no point named HOME (home plate is the origin)")
    lon0, lat0 = by_name["HOME"].lonlat[0]
    for f in feats:
        f.xy = [to_local(lo, la, lon0, lat0) for lo, la in f.lonlat]

    def pt(n):
        f = by_name.get(n)
        return f.xy[0] if f and f.kind == "point" else None

    # Orientation: CF first, then 2B, then the bisector of the foul poles
    orient, source = None, None
    if pt("CF"):
        orient, source = bearing_deg(*pt("CF")), "HOME->CF"
    elif pt("2B"):
        orient, source = bearing_deg(*pt("2B")), "HOME->2B"
    elif pt("LF_POLE") and pt("RF_POLE"):
        bl, br = bearing_deg(*pt("LF_POLE")), bearing_deg(*pt("RF_POLE"))
        orient, source = (bl + ang_diff(br, bl) / 2) % 360, "foul-pole bisector"

    checks = []
    for n, ref in (official_dist or {}).items():
        p = pt(n)
        if p:
            d = math.hypot(*p)
            err = (d - ref) / ref * 100
            checks.append((n, d, ref, err, abs(err) <= scale_tol_pct))

    foul = None
    if pt("LF_POLE") and pt("RF_POLE"):
        bl, br = bearing_deg(*pt("LF_POLE")), bearing_deg(*pt("RF_POLE"))
        ang = abs(ang_diff(br, bl))
        bis = (bl + ang_diff(br, bl) / 2) % 360
        foul = {"angle": ang, "bisector": bis,
                "ok": abs(ang - 90) <= foul_tol_deg,
                "cf_vs_bisector": ang_diff(orient, bis) if orient is not None else None}

    return {"path": path, "feats": feats, "origin": (lat0, lon0),
            "orient": orient, "orient_src": source, "checks": checks, "foul": foul,
            "example": any(f.example for f in feats)}


def build_geometry(res: dict, grid_n: int = 3, delta: float = 0.0) -> dict:
    """Model input: local coordinates of every feature and the sample grid of every section.

    delta rotates the whole stadium clockwise (degrees). Coordinates are rounded to 1 mm.
    """
    feats, b = res["feats"], res["orient"]
    lat0, lon0 = res["origin"]
    samples = {}
    for f in feats:
        if f.kind == "polygon" and f.name.upper().startswith("SEC_"):
            if len(f.xy) == 4:
                samples[f.name] = grid_points(f.xy, grid_n)
            else:
                warnings.warn(f"{f.name}: {len(f.xy)} vertices, expected 4; no sample points "
                              "(redraw the section with 4 corners)")

    rot = (lambda p: rotate_cw(p[0], p[1], delta)) if delta else (lambda p: p)

    def r2(p):
        return [round(c, 3) for c in rot(p)]

    return {
        "source_file": res["path"].name,
        "example_data": res["example"],
        "origin_home_plate": {"lat": lat0, "lon": lon0},
        "frame": "x=east (m), y=north (m), origin=home plate, true north",
        "orientation_deg": None if b is None else round((b + delta) % 360, 3),
        "rotation_applied_deg": delta,
        "grid_n": grid_n,
        "points": {f.name: r2(f.xy[0]) for f in feats if f.kind == "point"},
        "polygons": {f.name: [r2(p) for p in f.xy] for f in feats if f.kind == "polygon"},
        "lines": {f.name: [r2(p) for p in f.xy] for f in feats if f.kind == "line"},
        "sample_points_xy": {s: [r2(p) for p in pts] for s, pts in samples.items()},
    }


def checks_table(res: dict) -> pd.DataFrame:
    """Orientation, scale checks and foul-line angle as one table."""
    rows = []
    if res["orient"] is not None:
        rows.append(dict(check="orientation (home -> centre field)", measured=res["orient"],
                         reference=np.nan, error=np.nan, unit="deg", passed=""))
    for n, d, ref, err, ok in res["checks"]:
        rows.append(dict(check=f"distance HOME -> {n}", measured=d, reference=ref,
                         error=err, unit="m (error in %)", passed=bool(ok)))
    if res["foul"]:
        f = res["foul"]
        rows.append(dict(check="angle between the foul lines", measured=f["angle"], reference=90.0,
                         error=f["angle"] - 90.0, unit="deg", passed=bool(f["ok"])))
        if f["cf_vs_bisector"] is not None:
            rows.append(dict(check="centre-field bearing minus foul-line bisector",
                             measured=f["cf_vs_bisector"], reference=0.0,
                             error=f["cf_vs_bisector"], unit="deg", passed=""))
    return pd.DataFrame(rows).round(3)


def sections_table(res: dict) -> pd.DataFrame:
    """Area, centroid and stadium side of every roof and section polygon."""
    b = res["orient"]
    rows = []
    for f in res["feats"]:
        if f.kind != "polygon" or not f.name.upper().startswith(("SEC_", "ROOF_")):
            continue
        area, (cx, cy) = shoelace(f.xy)
        side, phi = "", np.nan
        if b is not None:
            u, v = to_stadium_frame(cx, cy, b)
            phi = math.degrees(math.atan2(u, v))
            side = side_label(phi)
        rows.append(dict(name=f.name, vertices=len(f.xy), area_m2=area, centroid_x=cx,
                         centroid_y=cy, dist_from_home_m=math.hypot(cx, cy),
                         angle_from_cf_axis_deg=phi, side=side))
    return pd.DataFrame(rows).round(2)


def compare_orientations(results: list[dict], min_tol_deg: float = 2.0) -> dict:
    """Spread of the orientation between several digitisers' files.

    The suggested sensitivity range is the larger of that spread and min_tol_deg.
    """
    orients = [(r["path"].name, r["orient"]) for r in results if r["orient"] is not None]
    if len(orients) < 2:
        return {"orientations": orients, "spread_deg": None, "suggested_range_deg": min_tol_deg}
    ref = orients[0][1]
    devs = [ang_diff(o, ref) for _, o in orients]
    spread = max(devs) - min(devs)
    return {"orientations": orients, "spread_deg": spread,
            "suggested_range_deg": max(spread, min_tol_deg)}


# ---------------------------------------------------------------------------
# 2. Plane geometry for the scene: roof edge chain, normals, ear clipping
# ---------------------------------------------------------------------------
def mound_xy(orient_deg: float, distance: float) -> tuple[float, float]:
    """A point on the field (the pitcher's mound), used to tell the field side of a roof."""
    b = math.radians(orient_deg)
    return distance * math.sin(b), distance * math.cos(b)


def classify_inner(roofs: dict, mound) -> dict:
    """A roof vertex seen directly from the mound lies on the field-side (inner) edge."""
    polys = [Polygon(v) for v in roofs.values()]
    out = {}
    for name, ring in roofs.items():
        out[name] = [sum(LineString([mound, v]).intersection(p).length for p in polys) < 0.5
                     for v in ring]
    return out


def inner_run(ring, flags):
    """Indices of the consecutive inner-edge vertices of one roof ring."""
    n = len(ring)
    start = next(i for i in range(n) if flags[i] and not flags[(i - 1) % n])
    idx, i = [], start
    while flags[i % n] and len(idx) < n:
        idx.append(i % n)
        i += 1
    return idx


def build_chain(roofs, flags, chain_order, roof_side):
    """One inner-edge chain from the first-base end, behind home, to the third-base end."""
    pts, side, ring_ref = [], [], []
    for name in chain_order:
        idx = inner_run(roofs[name], flags[name])
        seg = [tuple(roofs[name][i]) for i in idx]
        refs = [(name, i) for i in idx]
        if pts and math.dist(pts[-1], seg[0]) > math.dist(pts[-1], seg[-1]):
            seg, refs = seg[::-1], refs[::-1]
        if pts and math.dist(pts[-1], seg[0]) < 1.5:      # drop the duplicated joint vertex
            ring_ref[-1] = ring_ref[-1] + [refs[0]]
            seg, refs = seg[1:], refs[1:]
        pts += seg
        side += [roof_side[name]] * len(seg)
        ring_ref += [[r] for r in refs]
    return np.array(pts), side, ring_ref


def chain_normals(pts, mound):
    """Unit normals of the chain, pointing towards the field."""
    n = len(pts)
    nrm = np.zeros_like(pts)
    for i in range(n):
        a, b = pts[max(i - 1, 0)], pts[min(i + 1, n - 1)]
        t = (b - a) / np.linalg.norm(b - a)
        v = np.array([-t[1], t[0]])
        if np.dot(v, np.array(mound) - pts[i]) < 0:
            v = -v
        nrm[i] = v
    return nrm


def ray_segment_t(p, d, a, b):
    """2D: parameter t where the ray p + t·d meets the segment ab (inf if it does not)."""
    e = b - a
    den = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(den) < 1e-12:
        return math.inf
    w = a - p
    t = (w[0] * (-e[1]) - w[1] * (-e[0])) / den
    s = (d[0] * w[1] - d[1] * w[0]) / den
    return t if (t > 0 and 0 <= s <= 1) else math.inf


def clip_profile(prof, dmax):
    """Cut a stand profile at depth dmax (the distance to the foul line) with a front wall."""
    if prof[-1][0] <= dmax:
        return prof
    out = []
    for (d0, z0), (d1, z1) in zip(prof[:-1], prof[1:]):
        if d1 <= dmax:
            out.append((d0, z0))
            continue
        out.append((d0, z0) if d0 <= dmax else (dmax, z0))
        zc = z0 + (z1 - z0) * ((dmax - d0) / (d1 - d0) if d1 > d0 else 0.0)
        while len(out) < len(prof) - 1:
            out.append((dmax, zc))
        out.append((dmax, 0.0))
        return out[: len(prof)]
    return out


def ear_clip(ring):
    """Ear clipping of a simple polygon into triangles.

    Works for the concave roof strips too: no triangle is created outside the polygon,
    so the triangle areas add up to the polygon area.
    """
    pts = [tuple(p) for p in ring]
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts))) / 2
    idx = list(range(len(pts)))
    if area < 0:
        idx.reverse()                                  # make the ring counter-clockwise

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def inside(p, a, b, c):
        return cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 1e-9:                 # reflex vertex: not an ear
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((a, b, c))
            idx.pop(k)
            break
        else:
            raise RuntimeError("ear clipping failed (self-intersecting polygon?)")
    tris.append(tuple(pts[i] for i in idx))
    return tris
