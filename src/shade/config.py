"""Reads data/config.yaml into one Config object.

Every setting in the YAML file is written as  {value: ..., source: "..."}  (or, for list
entries such as scenarios and variants, as an item with its own "source"). The loader
refuses a setting without a source note, so the file stays a complete record of where
each number comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

import yaml

from .model import Variant

ROOT = Path(__file__).resolve().parents[2]          # repository root
DEFAULT_CONFIG = ROOT / "data" / "config.yaml"


@dataclass(frozen=True)
class Scenario:
    key: str        # short name used in tables, e.g. "Apr"
    date: str       # ISO date, e.g. "2026-04-19"
    label: str      # name used in figures, e.g. "19 Apr"


@dataclass
class Config:
    root: Path
    site: dict                       # lat, lon, altitude, tz
    start: str
    end: str
    step_min: int
    scenarios: list
    regression_dates: list
    stadium_file: Path
    roof_height: float
    head_height: float
    foul_margin: float
    mound_distance: float
    chain_order: list                # roof polygon names, first base -> home -> third base
    roof_side: dict                  # roof polygon name -> stand profile name
    profiles: dict                   # profile name -> [(d, z), ...]
    grid_n: int
    variants: list
    check_step_min: int
    check_grid_n: int
    section_labels: dict
    official_dist: dict
    scale_tol_pct: float
    foul_tol_deg: float
    labels_file: Path
    calibration_dates: list
    test_dates: list
    use_single_labelled: bool
    worked_example: dict
    figures: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def window_min(self) -> int:
        """Length of the game window in minutes."""
        (h0, m0), (h1, m1) = (map(int, t.split(":")) for t in (self.start, self.end))
        return (h1 * 60 + m1) - (h0 * 60 + m0)

    def label(self, section: str) -> str:
        return self.section_labels.get(section, section)

    def scenario(self, key: str) -> Scenario:
        for s in list(self.scenarios) + list(self.regression_dates):
            if s.key == key:
                return s
        raise KeyError(f"no scenario with key {key!r}")


def _val(node, path: str):
    """Return the value of a {value, source} setting; complain if the source is missing."""
    if not isinstance(node, dict) or "value" not in node:
        raise ValueError(f"config: '{path}' must be written as {{value: ..., source: ...}}")
    if not str(node.get("source") or "").strip():
        raise ValueError(f"config: '{path}' has no source note")
    return node["value"]


def _items(nodes, path: str):
    for i, n in enumerate(nodes):
        if not str(n.get("source") or "").strip():
            raise ValueError(f"config: '{path}[{i}]' has no source note")
    return nodes


def _scenarios(nodes, path: str) -> list:
    out = []
    for n in _items(nodes, path):
        _date.fromisoformat(str(n["date"]))             # fails on a malformed date
        out.append(Scenario(str(n["key"]), str(n["date"]), str(n.get("label", n["key"]))))
    return out


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.resolve().parent.parent

    site, tw, geo = raw["site"], raw["time_window"], raw["geometry"]
    chain = _val(geo["roof_chain"], "geometry.roof_chain")
    profiles = {name: [tuple(map(float, p)) for p in _val(node, f"stand_profiles.{name}")]
                for name, node in raw["stand_profiles"].items()}
    val = raw["validation"]
    sc = raw["scale_checks"]

    cfg = Config(
        root=root,
        site=dict(lat=_val(site["latitude"], "site.latitude"),
                  lon=_val(site["longitude"], "site.longitude"),
                  altitude=_val(site["altitude_m"], "site.altitude_m"),
                  tz=_val(site["timezone"], "site.timezone")),
        start=str(_val(tw["start"], "time_window.start")),
        end=str(_val(tw["end"], "time_window.end")),
        step_min=int(_val(tw["step_min"], "time_window.step_min")),
        scenarios=_scenarios(raw["scenarios"], "scenarios"),
        regression_dates=_scenarios(raw["regression_dates"], "regression_dates"),
        stadium_file=root / _val(geo["stadium_file"], "geometry.stadium_file"),
        roof_height=float(_val(geo["roof_height_m"], "geometry.roof_height_m")),
        head_height=float(_val(geo["head_height_m"], "geometry.head_height_m")),
        foul_margin=float(_val(geo["foul_margin_m"], "geometry.foul_margin_m")),
        mound_distance=float(_val(geo["mound_distance_m"], "geometry.mound_distance_m")),
        chain_order=[c["roof"] for c in chain],
        roof_side={c["roof"]: str(c["profile"]) for c in chain},
        profiles=profiles,
        grid_n=int(_val(raw["sampling"]["grid_n"], "sampling.grid_n")),
        variants=[Variant(str(n["name"]), float(n["vscale"]), float(n["overhang"]),
                          float(n["orient_deg"])) for n in _items(raw["variants"], "variants")],
        check_step_min=int(_val(raw["resolution_checks"]["step_min"], "resolution_checks.step_min")),
        check_grid_n=int(_val(raw["resolution_checks"]["grid_n"], "resolution_checks.grid_n")),
        section_labels=dict(_val(raw["section_labels"], "section_labels")),
        official_dist={str(k): float(v) for k, v in
                       _val(sc["distances_m"], "scale_checks.distances_m").items()},
        scale_tol_pct=float(_val(sc["distance_tolerance_pct"], "scale_checks.distance_tolerance_pct")),
        foul_tol_deg=float(_val(sc["foul_angle_tolerance_deg"], "scale_checks.foul_angle_tolerance_deg")),
        labels_file=root / _val(val["labels_file"], "validation.labels_file"),
        calibration_dates=[str(d) for d in _val(val["calibration_dates"], "validation.calibration_dates")],
        test_dates=[str(d) for d in _val(val["test_dates"], "validation.test_dates")],
        use_single_labelled=bool(_val(val["use_single_labelled"], "validation.use_single_labelled")),
        worked_example={k: _val(n, f"worked_example.{k}") for k, n in raw["worked_example"].items()},
        figures={k: _val(n, f"figures.{k}") for k, n in raw["figures"].items()},
        raw=raw,
    )
    _check(cfg)
    return cfg


def _check(cfg: Config) -> None:
    if cfg.window_min <= 0:
        raise ValueError("config: time_window.end must be later than time_window.start")
    for step in (cfg.step_min, cfg.check_step_min):
        if cfg.window_min % step:
            raise ValueError(f"config: a {step}-minute step does not divide the {cfg.window_min}-minute window")
    lengths = {len(p) for p in cfg.profiles.values()}
    if len(lengths) != 1:
        raise ValueError("config: every stand profile needs the same number of (d, z) points")
    for roof, prof in cfg.roof_side.items():
        if prof not in cfg.profiles:
            raise ValueError(f"config: roof {roof} uses the unknown stand profile {prof!r}")
    if cfg.variants[0].name != "base":
        raise ValueError("config: the first variant must be 'base'")
    names = [v.name for v in cfg.variants]
    if len(set(names)) != len(names):
        raise ValueError("config: variant names must be unique")
    both = set(cfg.calibration_dates) & set(cfg.test_dates)
    if both:
        raise ValueError(f"config: dates in both the calibration and the test list: {sorted(both)}")
