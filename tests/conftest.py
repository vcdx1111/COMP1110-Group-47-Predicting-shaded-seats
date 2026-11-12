"""Shared fixtures: the config, the digitised stadium and the base scene."""
from pathlib import Path

import pytest

from shade.config import load_config
from shade.geometry import analyse, build_geometry
from shade.model import build_scene

REFERENCE = Path(__file__).resolve().parent / "reference"


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def res(cfg):
    return analyse(cfg.stadium_file, cfg.official_dist, cfg.scale_tol_pct, cfg.foul_tol_deg)


@pytest.fixture(scope="session")
def geom(cfg, res):
    return build_geometry(res, cfg.grid_n)


@pytest.fixture(scope="session")
def base_scene(cfg, geom):
    return build_scene(geom, cfg.variants[0], cfg)


@pytest.fixture(scope="session")
def reference_dir():
    return REFERENCE
