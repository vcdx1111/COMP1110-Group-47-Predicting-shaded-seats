"""Sun position and the direction vector from a seat towards the sun.

Sun angles come from pvlib (NREL Solar Position Algorithm). Azimuth is measured clockwise
from true north, the same convention as the stadium coordinates (x = east, y = north).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def sun_positions(times: pd.DatetimeIndex, site: dict) -> pd.DataFrame:
    """Apparent elevation and azimuth (degrees) at the given timezone-aware times."""
    sp = pvlib.solarposition.get_solarposition(
        times, site["lat"], site["lon"], altitude=site["altitude"])
    return pd.DataFrame({"elev": sp["apparent_elevation"].values, "az": sp["azimuth"].values},
                        index=times)


def sun_table(date: str, site: dict, start: str, end: str, step_min: int) -> pd.DataFrame:
    """Sun position at the start of every step in [start, end) on one date.

    With 14:00-17:00 and 5-minute steps this gives 36 rows (14:00 ... 16:55), and each row
    stands for the 5 minutes that follow it, so the exposure adds up to 180 minutes.
    """
    times = pd.date_range(f"{date} {start}", f"{date} {end}", freq=f"{step_min}min",
                          tz=site["tz"], inclusive="left")
    return sun_positions(times, site)


def sun_direction(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector from the seat towards the sun: (east, north, up)."""
    a, A = np.radians(elevation_deg), np.radians(azimuth_deg)
    return np.array([np.cos(a) * np.sin(A), np.cos(a) * np.cos(A), np.sin(a)])


def sun_vectors(sun: pd.DataFrame, orient_delta: float = 0.0) -> np.ndarray:
    """Direction vectors (T,3) for a sun table.

    Turning the stadium by +δ degrees is the same as turning the sun azimuth by -δ.
    """
    a = np.radians(sun["elev"].values)
    A = np.radians(sun["az"].values - orient_delta)
    return np.c_[np.cos(a) * np.sin(A), np.cos(a) * np.cos(A), np.sin(a)]
