"""The config file: complete, documented, and consistent with the project's scenario rules."""
from datetime import date

import pytest
import yaml

from shade.config import DEFAULT_CONFIG, load_config
from shade.sun import sun_table


def test_scenarios_are_sundays_at_1400(cfg):
    assert [s.date for s in cfg.scenarios] == ["2026-04-19", "2026-05-24", "2026-09-20"]
    for s in cfg.scenarios:
        assert date.fromisoformat(s.date).weekday() == 6, f"{s.date} is not a Sunday"
    assert (cfg.start, cfg.end, cfg.step_min) == ("14:00", "17:00", 5)


def test_time_steps_cover_the_window(cfg):
    for step, n in [(cfg.step_min, 36), (cfg.check_step_min, 18)]:
        sun = sun_table(cfg.scenarios[0].date, cfg.site, cfg.start, cfg.end, step)
        assert len(sun) == n
        assert len(sun) * step == cfg.window_min == 180
        assert sun.index[0].strftime("%H:%M") == "14:00"


def test_dimensions_from_the_plan(cfg):
    assert cfg.roof_height == 22.5
    assert cfg.head_height == 1.2
    assert [v.name for v in cfg.variants] == ["base", "height -10%", "height +10%", "overhang -10%",
                                              "overhang +10%", "orient -2°", "orient +2°"]
    assert (cfg.grid_n, cfg.check_grid_n, cfg.check_step_min) == (3, 5, 10)


def test_setting_without_a_source_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    del raw["geometry"]["roof_height_m"]["source"]
    (tmp_path / "data").mkdir()
    bad = tmp_path / "data" / "config.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="roof_height_m"):
        load_config(bad)


def test_date_in_both_validation_lists_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["validation"]["calibration_dates"]["value"] = ["2026-04-19"]
    raw["validation"]["test_dates"]["value"] = ["2026-04-19"]
    (tmp_path / "data").mkdir()
    bad = tmp_path / "data" / "config.yaml"
    bad.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="both"):
        load_config(bad)
