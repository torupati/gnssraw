"""Relative positioning CLI with fixed integer double-difference ambiguities."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

from app.gnss.coordinates import ecef_to_llh
from app.gnss.database import Epoch, GnssDatabase
from app.gnss.plot.position_plot import (
    plot_enu_time_series,
    plot_track_on_background_maps,
)
from app.gnss.relative_positioning import (
    RelativePositionEpochResult,
    Wgs84Position,
    write_relative_position_results_csv,
)


def parse_datetime(value: str) -> dt.datetime:
    """Parse datetime from ISO-like string."""
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime format: {value}. Use YYYY-MM-DD HH:MM:SS") from exc


def block_doble_differencing_matrix(
    sat_blocks: tuple[list[str], ...],
) -> tuple[np.ndarray, list[str]]:
    """Create one global DD matrix by stacking block-wise DD rows."""
    flat_sat_order = [sat for block in sat_blocks for sat in block]
    sat_index = {sat: i for i, sat in enumerate(flat_sat_order)}
    m_total = len(flat_sat_order)
    rows: list[np.ndarray] = []

    for block in sat_blocks:
        if len(block) < 2:
            continue
        ref_idx = sat_index[block[0]]
        for sat in block[1:]:
            sat_idx = sat_index[sat]
            dd_row = np.zeros(2 * m_total, dtype=int)
            dd_row[ref_idx] = 1
            dd_row[sat_idx] = -1
            dd_row[m_total + ref_idx] = -1
            dd_row[m_total + sat_idx] = 1
            rows.append(dd_row)

    if not rows:
        return np.zeros((0, 2 * m_total), dtype=int), flat_sat_order
    return np.vstack(rows), flat_sat_order


def load_epoch_data(
    db_path: Path,
    epoch_dt: dt.datetime,
    sat_ids: list[str],
    bands: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]], np.ndarray | None]:
    """Load satellite positions and selected-band observations for one epoch."""
    db = GnssDatabase(db_path)
    satpos: dict[str, np.ndarray] = {}
    obs: dict[str, dict[str, dict[str, float]]] = {}
    spp_pos: np.ndarray | None = None

    session = db.Session()
    try:
        epoch = session.query(Epoch).filter_by(datetime=epoch_dt).one()

        if epoch.spp_solution is not None:
            spp_pos = np.array(
                [epoch.spp_solution.x, epoch.spp_solution.y, epoch.spp_solution.z],
                dtype=float,
            )

        for sat in epoch.satellites:
            sid = sat.satellite_id
            if sid not in sat_ids:
                continue
            if sat.position is not None:
                satpos[sid] = np.array([sat.position.x, sat.position.y, sat.position.z], dtype=float)
            obs[sid] = {
                sig.band: {
                    "code_range": float(sig.pseudorange),
                    "phase_range": float(sig.carrier_phase),
                }
                for sig in sat.signals
                if sig.band in bands
            }
    finally:
        session.close()

    return satpos, obs, spp_pos


def list_epochs_between(db_path: Path, start: dt.datetime, end: dt.datetime) -> list[dt.datetime]:
    """List epoch datetimes within [start, end]."""
    db = GnssDatabase(db_path)
    session = db.Session()
    try:
        rows = session.query(Epoch.datetime).filter(Epoch.datetime >= start, Epoch.datetime <= end).order_by(Epoch.datetime).all()
        return [r[0] for r in rows]
    finally:
        session.close()


def check_epoch_has_all_required(
    satpos_rover: dict[str, np.ndarray],
    satpos_base: dict[str, np.ndarray],
    obs_rover: dict[str, dict[str, dict[str, float]]],
    obs_base: dict[str, dict[str, dict[str, float]]],
    sat_order: list[str],
    bands: list[str],
) -> tuple[bool, str]:
    """Validate required satellites and bands exist for both receivers."""
    for sat in sat_order:
        if sat not in satpos_rover:
            return False, f"missing rover satellite position: {sat}"
        if sat not in satpos_base:
            return False, f"missing base satellite position: {sat}"
        if sat not in obs_rover:
            return False, f"missing rover observation: {sat}"
        if sat not in obs_base:
            return False, f"missing base observation: {sat}"
        for b in bands:
            if b not in obs_rover[sat]:
                return False, f"missing rover band {b}: {sat}"
            if b not in obs_base[sat]:
                return False, f"missing base band {b}: {sat}"
    return True, ""


def estimate_position_with_fixed_ambiguity(
    satpos_rover: dict[str, np.ndarray],
    satpos_base: dict[str, np.ndarray],
    sat_order: list[str],
    dd_matrix: np.ndarray,
    bands: list[str],
    wavelengths: dict[str, float],
    sigma_phi: dict[str, float],
    dd_obs_by_band: dict[str, np.ndarray],
    dd_ambiguity_by_band: dict[str, np.ndarray],
    base_pos: np.ndarray,
    initial_pos: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Estimate rover position for one epoch with fixed integer ambiguities.
    Args:
        satpos_rover: Satellite positions for the rover receiver.
        satpos_base: Satellite positions for the base receiver.
        sat_order: List of satellite identifiers.
        dd_matrix: Double-differenced observation matrix.
        bands: List of frequency bands.
        wavelengths: Dictionary of wavelengths for each band.
        sigma_phi: Dictionary of phase noise standard deviations for each band.
        dd_obs_by_band: Double-differenced observations by band.
        dd_ambiguity_by_band: Double-differenced ambiguities by band.
        base_pos: Position of the base receiver.
        initial_pos: Initial position estimate for the rover.

    Returns:
        Tuple containing the estimated rover position, number of iterations used, and residual norm.
    """
    num_dd = dd_matrix.shape[0]
    num_bands = len(bands)
    n_obs = num_dd * num_bands
    m = len(sat_order)
    d_left = dd_matrix[:, :m]

    c_dd_base = dd_matrix @ dd_matrix.T
    w = np.zeros((n_obs, n_obs), dtype=float)
    for ib, b in enumerate(bands):
        s = sigma_phi.get(b, 0.01)
        c_block = (s**2) * c_dd_base
        w_block = np.linalg.inv(c_block)
        row = ib * num_dd
        w[row : row + num_dd, row : row + num_dd] = w_block

    x = initial_pos.copy()
    used_itr = 0
    residual_norm = float("nan")

    for itr in range(10):
        dy = np.zeros(n_obs, dtype=float)
        h = np.zeros((n_obs, 3), dtype=float)

        for ib, b in enumerate(bands):
            wl = wavelengths[b]
            row = ib * num_dd

            ranges = [np.linalg.norm(satpos_rover[s] - x) for s in sat_order]
            ranges += [np.linalg.norm(satpos_base[s] - base_pos) for s in sat_order]
            dd_rho = dd_matrix @ np.array(ranges, dtype=float)

            los = np.array(
                [(satpos_rover[s] - x) / np.linalg.norm(satpos_rover[s] - x) for s in sat_order],
                dtype=float,
            )

            dy[row : row + num_dd] = dd_obs_by_band[b] - (dd_rho / wl + dd_ambiguity_by_band[b])
            h[row : row + num_dd, :] = -d_left @ los / wl

        htw = h.T @ w
        dx, *_ = np.linalg.lstsq(htw @ h, htw @ dy, rcond=None)
        x = x + dx
        used_itr = itr + 1
        residual_norm = float(np.linalg.norm(dy)) / len(dy)

        if np.linalg.norm(dx) < 1e-5:
            break

    return x, used_itr, residual_norm


def main() -> int:
    parser = argparse.ArgumentParser(description="Relative positioning with fixed DD integer ambiguities")
    parser.add_argument("--base-db", required=True, help="Base station SQLite DB")
    parser.add_argument("--rover-db", required=True, help="Rover station SQLite DB")
    parser.add_argument(
        "--ambiguity-json",
        required=True,
        help="JSON file containing DD ambiguity information",
    )
    parser.add_argument("--start", required=True, type=parse_datetime, help="Start time")
    parser.add_argument("--end", required=True, type=parse_datetime, help="End time")
    parser.add_argument(
        "--out",
        default="rp_bias_results.json",
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--out-csv",
        default="rp_bias_results.csv",
        help="Output CSV path for positioning summary",
    )
    parser.add_argument(
        "--plot-enu",
        action="store_true",
        help="Plot ENU time series (3 rows: E/N/U) with base station as origin",
    )
    parser.add_argument(
        "--out-plot",
        default="rp_bias_enu.png",
        help="Output image path for ENU time-series plot",
    )
    parser.add_argument(
        "--out-map-dir",
        default="rp_bias_map_plots",
        help="Output directory for 2D map plots (ESRI/OSM/GSI)",
    )
    args = parser.parse_args()

    base_db = Path(args.base_db)
    rover_db = Path(args.rover_db)
    ambiguity_json = Path(args.ambiguity_json)
    out_path = Path(args.out)
    out_csv_path = Path(args.out_csv)
    out_plot_path = Path(args.out_plot)
    out_map_dir = Path(args.out_map_dir)

    if not base_db.exists():
        raise FileNotFoundError(f"Base DB not found: {base_db}")
    if not rover_db.exists():
        raise FileNotFoundError(f"Rover DB not found: {rover_db}")
    if not ambiguity_json.exists():
        raise FileNotFoundError(f"Ambiguity JSON not found: {ambiguity_json}")

    with ambiguity_json.open("r", encoding="utf-8") as f:
        amb_info = json.load(f)

    bands = list(amb_info["bands"])
    satellite_blocks = tuple([list(block) for block in amb_info["satellite_blocks"]])
    dd_ambiguity_by_band = {b: np.array(amb_info["dd_phase_biases_int"][b], dtype=float) for b in bands}
    wavelengths = {b: float(amb_info["wavelengths"][b]) for b in bands}
    sigma_phi = {b: float(amb_info.get("sigma_phi", {}).get(b, 0.01)) for b in bands}

    dd_matrix, sat_order = block_doble_differencing_matrix(satellite_blocks)
    if dd_matrix.shape[0] == 0:
        raise ValueError("No valid DD rows. Check satellite_blocks in ambiguity JSON.")

    base_pos_list = amb_info.get("base_station_position_ecef", amb_info.get("base_position"))
    if base_pos_list is None:
        raise ValueError("Ambiguity JSON must contain base_station_position_ecef (or base_position).")
    base_pos = np.array(base_pos_list, dtype=float)
    if base_pos.shape != (3,):
        raise ValueError("Base station position in JSON must be 3-element ECEF [x, y, z].")

    ref_rover_pos = np.array(amb_info.get("reference_rover_position", [0.0, 0.0, 0.0]), dtype=float)

    base_epochs = set(list_epochs_between(base_db, args.start, args.end))
    rover_epochs = set(list_epochs_between(rover_db, args.start, args.end))
    common_epochs = sorted(base_epochs & rover_epochs)

    if not common_epochs:
        raise ValueError("No common epochs found in the requested interval.")

    print(f"Found {len(common_epochs)} common epochs in range.")
    print(f"satellite blocks: {satellite_blocks}")
    print(f"bands: {bands}")

    base_llh = ecef_to_llh(base_pos)

    epoch_results: list[RelativePositionEpochResult] = []
    results = {
        "base_db": str(base_db),
        "rover_db": str(rover_db),
        "ambiguity_json": str(ambiguity_json),
        "start": args.start.isoformat(sep=" "),
        "end": args.end.isoformat(sep=" "),
        "base_pos": base_pos.tolist(),
        "satellite_blocks": [list(block) for block in satellite_blocks],
        "bands": bands,
        "epochs": [],
    }

    for epoch_dt in common_epochs:
        satpos_rover, obs_rover, spp_rover = load_epoch_data(rover_db, epoch_dt, sat_order, bands)
        satpos_base, obs_base, _ = load_epoch_data(base_db, epoch_dt, sat_order, bands)

        valid, reason = check_epoch_has_all_required(satpos_rover, satpos_base, obs_rover, obs_base, sat_order, bands)
        if not valid:
            print(f"[{epoch_dt}] skipped: {reason}")
            continue

        dd_obs_by_band: dict[str, np.ndarray] = {}
        for b in bands:
            phi = [obs_rover[s][b]["phase_range"] for s in sat_order]
            phi += [obs_base[s][b]["phase_range"] for s in sat_order]
            dd_obs_by_band[b] = dd_matrix @ np.array(phi, dtype=float)

        init_pos = spp_rover if spp_rover is not None else ref_rover_pos
        est_pos, iters, res_norm = estimate_position_with_fixed_ambiguity(
            satpos_rover=satpos_rover,
            satpos_base=satpos_base,
            sat_order=sat_order,
            dd_matrix=dd_matrix,
            bands=bands,
            wavelengths=wavelengths,
            sigma_phi=sigma_phi,
            dd_obs_by_band=dd_obs_by_band,
            dd_ambiguity_by_band=dd_ambiguity_by_band,
            base_pos=base_pos,
            initial_pos=init_pos,
        )

        baseline = float(np.linalg.norm(est_pos - base_pos))
        rover_llh = ecef_to_llh(est_pos)
        num_sats = len(sat_order)
        print(f"[{epoch_dt}] pos={est_pos} baseline={baseline:.3f} m iters={iters} residual_norm={res_norm:.6e}")

        epoch_result = RelativePositionEpochResult(
            datetime=epoch_dt,
            rover_position_ecef=est_pos.tolist(),
            rover_position_wgs84=Wgs84Position(
                lat_deg=float(rover_llh[0]),
                lon_deg=float(rover_llh[1]),
                height_m=float(rover_llh[2]),
            ),
            base_position_ecef=base_pos.tolist(),
            base_position_wgs84=Wgs84Position(
                lat_deg=float(base_llh[0]),
                lon_deg=float(base_llh[1]),
                height_m=float(base_llh[2]),
            ),
            num_satellites=num_sats,
            baseline_length=baseline,
            iterations=iters,
            residual_norm=res_norm,
        )
        epoch_results.append(epoch_result)
        results["epochs"].append(epoch_result.to_json_dict())

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_relative_position_results_csv(epoch_results, out_csv_path)

    if args.plot_enu:
        plot_enu_time_series(epoch_results, out_plot_path)

    map_paths = plot_track_on_background_maps(epoch_results, out_map_dir)

    print(f"Saved result JSON: {out_path}")
    print(f"Saved result CSV : {out_csv_path}")
    if args.plot_enu:
        print(f"Saved ENU plot   : {out_plot_path}")
    if map_paths:
        print(f"Saved map plots  : {len(map_paths)} files in {out_map_dir}")
    else:
        print("Saved map plots  : none (all providers failed or no data)")
    print(f"Solved epochs: {len(results['epochs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
