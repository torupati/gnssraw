"""Tropospheric delay model for GNSS signal propagation."""

from __future__ import annotations

from logging import getLogger

import numpy as np

from app.gnss.coordinates import ecef_to_enu_matrix

logger = getLogger(__name__)


def tropospheric_delay(receiver_llh: np.ndarray, sat_pos: np.ndarray, recv_pos: np.ndarray) -> float:
    """
    Calculate tropospheric delay correction for the signal path.

    Estimates the delay of GNSS signals propagating through the troposphere
    using a simplified model based on receiver altitude, elevation angle,
    and standard atmospheric parameters.

    Args:
        receiver_llh: Receiver position in geodetic coordinates [lat, lon, h]
                     (degrees, degrees, meters)
        sat_pos: Satellite position in ECEF coordinates (meters)
        recv_pos: Receiver position in ECEF coordinates (meters)

    Returns:
        Tropospheric delay in meters. Returns 0.0 if:
        - Receiver position is invalid (norm < 1.0 m)
        - Altitude is outside valid range (-1000 to 100000 m)
        - Satellite elevation angle is below 5 degrees
        - Atmospheric calculation would be invalid
    """
    if np.linalg.norm(recv_pos) < 1.0:
        return 0.0

    lat, lon, h = receiver_llh
    if not np.isfinite(h) or h < -1000.0 or h > 1e5:
        return 0.0
    enu = ecef_to_enu_matrix(lat, lon) @ (sat_pos - recv_pos)
    east, north, up = enu
    horiz = np.hypot(east, north)
    elev = np.arctan2(up, horiz)
    if elev <= np.radians(5.0):
        return 0.0

    base = 1 - 2.2557e-5 * h
    logger.debug("Base: %s, Elev: %.2f deg", base, np.degrees(elev))
    if base <= 0:
        return 0.0
    pressure = 1013.25 * (base**5.2568)
    temperature = 291.15 - 0.0065 * h
    vapor_pressure = 11.7

    zenith_delay = 0.002277 * (pressure + (1255 / temperature + 0.05) * vapor_pressure)
    return zenith_delay / np.sin(elev)
