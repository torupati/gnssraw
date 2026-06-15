"""Single Point Positioning (SPP) using RINEX observation and navigation files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger, basicConfig, INFO
from pathlib import Path

import numpy as np

from app.gnss.constants import CLIGHT
from app.gnss.coordinates import ecef_to_enu_matrix, ecef_to_llh
from app.gnss.troposphere import tropospheric_delay
from app.gnss.ephemeris import (
    read_rinex_nav,
    compute_satellite_state,
    GPSEphemeris,
    datetime_to_gps_week_seconds,
    OMEGA_E,
)
from app.gnss.satellite_signals import (
    EpochObservations,
    SatelliteObservation,
    parse_rinex_observation_file,
)
from app.gnss.ionosphere import KlobucharManager, KlobucharModel
from app.gnss.database import GnssDatabase

logger = getLogger(__name__)


@dataclass
class SppSolution:
    datetime: datetime
    position_ecef: np.ndarray
    position_llh: np.ndarray
    clock_bias_m: float
    num_sats: int
    residuals: np.ndarray


def select_ephemeris(
    nav_data: dict[str, list[GPSEphemeris]],
    sv: str,
    sow: float,
) -> GPSEphemeris:
    """Select the best ephemeris message for a given satellite and time."""
    messages = nav_data.get(sv, [])
    if not messages:
        raise ValueError(f"No ephemeris available for satellite {sv}")
    # Return the ephemeris with the closest toe to sow
    return min(
        messages, key=lambda msg: abs((sow - msg.toe + 302400) % 604800 - 302400)
    )


def apply_earth_rotation_correction(
    sat_pos: np.ndarray, travel_time: float
) -> np.ndarray:
    """
    Apply Earth rotation correction to satellite position.

    Corrects the satellite position in ECEF coordinates to account for
    Earth's rotation during signal travel time. This is necessary because
    the satellite coordinates at transmission time must be rotated to match
    the Earth's orientation at reception time.

    Args:
        sat_pos: Satellite position in ECEF coordinates at transmission time
                (meters, 3-element array [x, y, z])
        travel_time: Signal travel time from satellite to receiver (seconds)

    Returns:
        Corrected satellite position in ECEF coordinates (meters)
    """
    theta = OMEGA_E * travel_time
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rotation = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    return rotation @ sat_pos


def collect_measurements(
    epoch: EpochObservations,
    nav_data: dict[str, list[GPSEphemeris]],
    sow: float,
) -> list[tuple[str, np.ndarray, float, SatelliteObservation]]:
    """
    Collect valid GPS measurements for an epoch.

    Iterates over satellites in the epoch, selects the best ephemeris,
    and computes satellite position and clock bias for each valid observation.

    Args:
        epoch: Observation data for a single epoch.
        nav_data: Broadcast ephemeris keyed by satellite ID.
        sow: GPS seconds of week for the epoch.

    Returns:
        List of (sat_id, sat_pos_ecef, clock_bias_s, sat_obs) tuples.
    """
    measurements = []
    for sat_id, sat_obs in epoch.iter_satellites():
        if not sat_id.startswith("G"):
            continue
        if "L1" not in sat_obs.signals:
            continue
        pr = sat_obs.signals["L1"].pseudorange
        if not np.isfinite(pr):
            continue

        try:
            nav = select_ephemeris(nav_data, sat_id, sow)
        except ValueError:
            continue

        try:
            sat_pos, dtsv = compute_satellite_state(nav, epoch.datetime, pr)
        except Exception as e:
            logger.warning("Failed to compute satellite state for %s: %s", sat_id, e)
            continue
        if not np.all(np.isfinite(sat_pos)) or not np.isfinite(dtsv):
            continue
        logger.debug(
            "  %s at %s: pr=%.3f m sat_pos=%s dtsv=%s s",
            sat_id,
            epoch.datetime.isoformat(),
            pr,
            sat_pos,
            dtsv,
        )
        measurements.append((sat_id, sat_pos, dtsv, sat_obs))
    return measurements


def build_observation_matrix(
    measurements: list[tuple[str, np.ndarray, float, SatelliteObservation]],
    receiver_pos: np.ndarray,
    receiver_clock: float,
    epoch_dt: datetime,
    ionosphere_manager: KlobucharManager,
    elevation_mask_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the design matrix H and residual vector v for least-squares positioning.

    For each measurement, applies earth rotation correction, tropospheric delay,
    and ionospheric delay to form the linearized observation equations.

    Args:
        measurements: List of (sat_id, sat_pos_ecef, clock_bias_s, sat_obs).
        receiver_pos: Current receiver position estimate in ECEF (meters).
        receiver_clock: Current receiver clock bias estimate (meters).
        epoch_dt: Epoch datetime for ionospheric model lookup.
        ionosphere_manager: Ionospheric correction manager.
        elevation_mask_deg: Minimum satellite elevation angle in degrees.
                           Satellites below this angle are excluded.

    Returns:
        Tuple of (H_mat, v_vec) where H_mat has shape (n, 4) and v_vec has shape (n,).
        Returns empty arrays if fewer than 1 valid observation.
    """
    elev_mask_rad = np.radians(elevation_mask_deg)
    H_rows = []
    v_rows = []
    receiver_llh = ecef_to_llh(receiver_pos)
    enu_rotation = ecef_to_enu_matrix(receiver_llh[0], receiver_llh[1])
    receiver_llh_rad = np.array(
        [
            np.radians(receiver_llh[0]),
            np.radians(receiver_llh[1]),
            receiver_llh[2],
        ]
    )
    iono_model = ionosphere_manager.get_model_for_time(epoch_dt)

    for sat_id, sat_pos, dtsv, sat_obs in measurements:
        pr = sat_obs.signals["L1"].pseudorange
        rho = np.linalg.norm(sat_pos - receiver_pos)
        if rho <= 0 or not np.isfinite(rho):
            continue
        travel_time = rho / CLIGHT
        sat_corr = apply_earth_rotation_correction(sat_pos, travel_time)
        if not np.all(np.isfinite(sat_corr)):
            continue
        rho = np.linalg.norm(sat_corr - receiver_pos)
        if rho <= 0 or not np.isfinite(rho):
            continue

        enu = enu_rotation @ (sat_corr - receiver_pos)
        east, north, up = enu
        horiz = np.hypot(east, north)
        elev = np.arctan2(up, horiz)
        az = np.arctan2(east, north)

        logger.debug(
            "  %s elev=%.1f deg az=%.1f deg %s",
            sat_id,
            np.degrees(elev),
            np.degrees(az),
            elevation_mask_deg,
        )
        if elevation_mask_deg > 0 and elev < elev_mask_rad:
            continue
        tropo = tropospheric_delay(receiver_llh, sat_corr, receiver_pos)
        if not np.isfinite(tropo):
            tropo = 0.0

        iono = 0.0
        if iono_model is not None:
            if elev > 0 and iono_model is not None:
                iono_val = iono_model.calculate_delay(
                    epoch_dt, receiver_llh_rad, az, elev
                )
                if np.isfinite(iono_val):
                    iono = iono_val
                    logger.debug(
                        "  iono %s at %s: %.3f m az=%.1f elev=%.1f deg",
                        sat_id,
                        epoch_dt.isoformat(),
                        iono,
                        np.degrees(az),
                        np.degrees(elev),
                    )

        predicted = rho + receiver_clock - CLIGHT * dtsv + tropo + iono
        v = pr - predicted
        if not np.isfinite(v):
            continue
        H = (receiver_pos - sat_corr) / rho
        if not np.all(np.isfinite(H)):
            continue
        H_rows.append([H[0], H[1], H[2], 1.0])
        v_rows.append(v)

    if not H_rows:
        return np.empty((0, 4)), np.empty(0)
    return np.array(H_rows), np.array(v_rows)


def single_point_positioning(
    epochs: list[EpochObservations],
    nav_data: dict[str, list[GPSEphemeris]],
    ionosphere_manager: KlobucharManager,
    max_iterations: int = 8,
    elevation_mask_deg: float = 20.0,
) -> list[SppSolution]:
    solutions: list[SppSolution] = []
    receiver_pos = np.zeros(3)
    receiver_clock = 0.0

    for epoch in epochs:
        _, sow = datetime_to_gps_week_seconds(epoch.datetime)
        measurements = collect_measurements(epoch, nav_data, sow)

        if len(measurements) < 4:
            solutions.append(
                SppSolution(
                    datetime=epoch.datetime,
                    position_ecef=receiver_pos.copy(),
                    position_llh=ecef_to_llh(receiver_pos),
                    clock_bias_m=receiver_clock,
                    num_sats=len(measurements),
                    residuals=np.array([]),
                )
            )
            continue

        last_H = None
        last_v = None
        last_dx = None
        for iteration in range(max_iterations):
            mask = elevation_mask_deg if iteration > 0 else 0.0
            H_mat, v_vec = build_observation_matrix(
                measurements,
                receiver_pos,
                receiver_clock,
                epoch.datetime,
                ionosphere_manager,
                elevation_mask_deg=mask,
            )

            if H_mat.shape[0] < 4:
                break

            dx, *_ = np.linalg.lstsq(H_mat, v_vec, rcond=None)
            last_H = H_mat
            last_v = v_vec
            last_dx = dx

            receiver_pos += dx[:3]
            receiver_clock += dx[3]

            if np.linalg.norm(dx[:3]) < 1e-3 and abs(dx[3]) < 1e-3:
                break

        if last_H is not None and last_v is not None and last_dx is not None:
            residuals = last_v - last_H @ last_dx
        else:
            residuals = np.array([])
        solutions.append(
            SppSolution(
                datetime=epoch.datetime,
                position_ecef=receiver_pos.copy(),
                position_llh=ecef_to_llh(receiver_pos),
                clock_bias_m=receiver_clock,
                num_sats=last_H.shape[0] if last_H is not None else 0,
                residuals=residuals,
            )
        )

    return solutions


def main() -> int:
    parser = argparse.ArgumentParser(description="Single Point Positioning (SPP)")
    parser.add_argument("rinex_obs", type=str, help="RINEX observation file")
    parser.add_argument("rinex_nav", type=str, help="RINEX navigation file")
    parser.add_argument(
        "--signal-code-map",
        type=str,
        default=str(Path(__file__).parent / ".signal_code_map.json"),
        help="Path to signal code map JSON",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="Limit number of epochs processed (0 = all)",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help="Path to SQLite database file to save observations and solutions",
    )

    args = parser.parse_args()
    basicConfig(level=INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    obs_path = Path(args.rinex_obs)
    nav_path = Path(args.rinex_nav)
    if not obs_path.exists() or not nav_path.exists():
        logger.error("Missing RINEX files: %s %s", obs_path, nav_path)
        return 1

    with Path(args.signal_code_map).open("r", encoding="utf-8") as f:
        signal_code_map = json.load(f)

    # Parse RINEX observation file and limit epochs if requested
    epochs = parse_rinex_observation_file(str(obs_path), signal_code_map)
    if args.max_epochs > 0:
        epochs = epochs[: args.max_epochs]

    # Parse RINEX navigation file to read ephemeris data
    nav_data, ion_params = read_rinex_nav(str(nav_path))

    logger.info("Ionosphere parameters from RINEX nav: %s", ion_params)

    ionosphere_manager = KlobucharManager()
    if "ion_alpha" in ion_params and "ion_beta" in ion_params:
        model = KlobucharModel(ion_params["ion_alpha"], ion_params["ion_beta"])

        # Determine valid time (using the first epoch's time or a placeholder)
        time_of_data = epochs[0].datetime if epochs else datetime.utcnow()
        ionosphere_manager.add_model(time_of_data, model)

    solutions = single_point_positioning(
        epochs, nav_data, ionosphere_manager=ionosphere_manager
    )

    for sol in solutions:
        lat, lon, h = sol.position_llh
        logger.info(
            "%s lat=%.8f lon=%.8f h=%.3f m clk=%.3f m sats=%d",
            sol.datetime.isoformat(),
            lat,
            lon,
            h,
            sol.clock_bias_m,
            sol.num_sats,
        )

    if args.database:
        db = GnssDatabase(args.database)
        db.save_epoch_observations(epochs)

        for epoch in epochs:
            _, sow = datetime_to_gps_week_seconds(epoch.datetime)
            measurements = collect_measurements(epoch, nav_data, sow)
            sat_positions: dict[str, dict[str, float | datetime | int]] = {}
            for sat_id, sat_pos, dtsv, _sat_obs in measurements:
                sat_positions[sat_id] = {
                    "datetime": epoch.datetime,
                    "nano_second": epoch.datetime.microsecond * 1000,
                    "x": float(sat_pos[0]),
                    "y": float(sat_pos[1]),
                    "z": float(sat_pos[2]),
                    "clock_bias": float(dtsv),
                }
            if sat_positions:
                db.save_satellite_positions(sat_positions, epoch.datetime)

        for sol in solutions:
            db.save_spp_solution(
                {
                    "position_ecef": sol.position_ecef.tolist(),
                    "position_llh": sol.position_llh.tolist(),
                    "clock_bias_m": sol.clock_bias_m,
                    "num_satellites": sol.num_sats,
                    "residuals": sol.residuals.tolist()
                    if sol.residuals.size > 0
                    else None,
                },
                sol.datetime,
            )
        logger.info("Saved observations and solutions to %s", args.database)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
