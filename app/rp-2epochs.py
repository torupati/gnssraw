"""Two-epoch relative positioning with double-differenced carrier phase."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.gnss.constants import CLIGHT, wlen_L1, wlen_L2, wlen_L5
from app.gnss.database import GnssDatabase, Epoch

freq_L1 = CLIGHT / wlen_L1
freq_L2 = CLIGHT / wlen_L2

WLENS = {
    "L1": wlen_L1,
    "L2": wlen_L2,
    "L5": wlen_L5,
    "WL": (wlen_L1 * wlen_L2) / (wlen_L2 - wlen_L1),
}


def parse_datetime(value: str) -> dt.datetime:
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime '{value}'. Use YYYY-MM-DD HH:MM:SS"
        ) from exc


def normalize_satellite_blocks(raw_blocks: Any) -> tuple[list[str], ...]:
    """Normalize satellite blocks to tuple[list[str], ...]."""
    if not isinstance(raw_blocks, (list, tuple)) or not raw_blocks:
        raise ValueError("satellite_blocks must be non-empty list/tuple")

    if all(isinstance(x, str) for x in raw_blocks):
        if len(raw_blocks) < 2:
            raise ValueError("satellite_blocks must contain at least 2 satellites")
        return (list(raw_blocks),)

    if all(isinstance(x, (list, tuple)) for x in raw_blocks):
        blocks: list[list[str]] = []
        for block in raw_blocks:
            if not all(isinstance(s, str) for s in block):
                raise ValueError("Each satellite ID must be str")
            if len(block) < 2:
                raise ValueError(
                    "Each satellite block must contain at least 2 satellites"
                )
            blocks.append(list(block))
        return tuple(blocks)

    raise ValueError("Invalid satellite_blocks format")


def add_widelane_to_epoch(obs_epoch: dict[str, dict[str, dict[str, float]]]) -> None:
    """Add wide-lane synthetic observation for each satellite if L1/L2 exist."""
    for obs in obs_epoch.values():
        if "L1" in obs and "L2" in obs:
            obs["WL"] = {
                "phase_range": obs["L1"]["phase_range"] - obs["L2"]["phase_range"],
                "code_range": (
                    freq_L1 * obs["L1"]["code_range"]
                    - freq_L2 * obs["L2"]["code_range"]
                )
                / (freq_L1 - freq_L2),
            }


def block_doble_differencing_matrix(
    sat_blocks: tuple[list[str], ...],
) -> tuple[np.ndarray, list[str]]:
    """Create one global DD matrix by stacking block-wise DD rows."""
    flat_sat_order = [sat for block in sat_blocks for sat in block]
    sat_index = {sat: i for i, sat in enumerate(flat_sat_order)}
    m_total = len(flat_sat_order)
    rows: list[np.ndarray] = []

    for block in sat_blocks:
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


def load_satpos_obs_and_spp(
    db_path: Path,
    epoch_dt: dt.datetime,
    sat_ids: list[str],
) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, dict[str, float]]], np.ndarray | None
]:
    """Load satellite positions, observations, and optional SPP position for one epoch."""
    db = GnssDatabase(db_path)
    satpos: dict[str, np.ndarray] = {}
    obs: dict[str, dict[str, dict[str, float]]] = {}
    spp_pos: np.ndarray | None = None

    session = db.Session()
    try:
        epoch_row = session.query(Epoch).filter_by(datetime=epoch_dt).one()
        if epoch_row.spp_solution is not None:
            spp_pos = np.array(
                [
                    epoch_row.spp_solution.x,
                    epoch_row.spp_solution.y,
                    epoch_row.spp_solution.z,
                ],
                dtype=float,
            )
        for sat in epoch_row.satellites:
            sid = sat.satellite_id
            if sid not in sat_ids:
                continue
            if sat.position is not None:
                satpos[sid] = np.array(
                    [sat.position.x, sat.position.y, sat.position.z], dtype=float
                )
            obs[sid] = {
                sig.band: {
                    "code_range": float(sig.pseudorange),
                    "phase_range": float(sig.carrier_phase),
                }
                for sig in sat.signals
            }
    finally:
        session.close()

    return satpos, obs, spp_pos


def dd_geometric_ranges(
    satpos_rover: dict[str, np.ndarray],
    satpos_base: dict[str, np.ndarray],
    sat_order: list[str],
    dd_matrix: np.ndarray,
    rover_pos: np.ndarray,
    base_pos: np.ndarray,
) -> np.ndarray:
    geom = [np.linalg.norm(satpos_rover[s] - rover_pos) for s in sat_order]
    geom += [np.linalg.norm(satpos_base[s] - base_pos) for s in sat_order]
    return dd_matrix @ np.array(geom, dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-epoch relative positioning")
    parser.add_argument("--base-db", required=True, help="Base station SQLite DB")
    parser.add_argument("--rover-db", required=True, help="Rover station SQLite DB")
    parser.add_argument(
        "--config-json",
        required=True,
        help="Config JSON path (start/end time, satellites, frequencies)",
    )
    parser.add_argument(
        "--out",
        default="rp_2epochs_result.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    base_db = Path(args.base_db)
    rover_db = Path(args.rover_db)
    config_path = Path(args.config_json)
    out_path = Path(args.out)

    if not base_db.exists():
        raise FileNotFoundError(f"Base DB not found: {base_db}")
    if not rover_db.exists():
        raise FileNotFoundError(f"Rover DB not found: {rover_db}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    try:
        epoch1 = parse_datetime(cfg["epochs"]["start_time"])
        epoch2 = parse_datetime(cfg["epochs"]["end_time"])
        bands = list(cfg["bands"])
        satellite_blocks = normalize_satellite_blocks(cfg["satellite_blocks"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"Invalid epoch format in config JSON: {e}")

    satellite_order = [sat for block in satellite_blocks for sat in block]
    for b in bands:
        if b not in WLENS:
            raise ValueError(f"Unsupported band: {b}")

    dd_matrix, satellite_order = block_doble_differencing_matrix(satellite_blocks)
    num_dd = dd_matrix.shape[0]
    if num_dd == 0:
        raise ValueError(
            "At least one satellite block must contain 2 or more satellites"
        )

    satpos_ep1_rover, obs_ep1_rover, spp_ep1_rover = load_satpos_obs_and_spp(
        rover_db, epoch1, satellite_order
    )
    satpos_ep2_rover, obs_ep2_rover, _ = load_satpos_obs_and_spp(
        rover_db, epoch2, satellite_order
    )
    satpos_ep1_base, obs_ep1_base, spp_ep1_base = load_satpos_obs_and_spp(
        base_db, epoch1, satellite_order
    )
    satpos_ep2_base, obs_ep2_base, _ = load_satpos_obs_and_spp(
        base_db, epoch2, satellite_order
    )

    if "WL" in bands:
        add_widelane_to_epoch(obs_ep1_rover)
        add_widelane_to_epoch(obs_ep2_rover)
        add_widelane_to_epoch(obs_ep1_base)
        add_widelane_to_epoch(obs_ep2_base)

    base_pos = (
        np.array(cfg.get("base_position_ecef"), dtype=float)
        if "base_position_ecef" in cfg
        else spp_ep1_base
    )
    if base_pos is None:
        raise ValueError(
            "base_position_ecef is not provided and base SPP solution is unavailable"
        )

    rover_pos = (
        np.array(cfg.get("initial_rover_position_ecef"), dtype=float)
        if "initial_rover_position_ecef" in cfg
        else spp_ep1_rover
    )
    if rover_pos is None:
        raise ValueError(
            "initial_rover_position_ecef is not provided and rover SPP solution is unavailable"
        )

    sigma_phi_cfg = cfg.get("sigma_phi", {})
    sigma_phi = {b: float(sigma_phi_cfg.get(b, 0.01)) for b in bands}

    dd_obs_ep1: dict[str, np.ndarray] = {}
    dd_obs_ep2: dict[str, np.ndarray] = {}
    dd_code_phase_bias_ep1: dict[str, np.ndarray] = {}
    dd_code_phase_bias_ep2: dict[str, np.ndarray] = {}

    for b in bands:
        wl = WLENS[b]
        phi1 = [obs_ep1_rover[s][b]["phase_range"] for s in satellite_order]
        phi1 += [obs_ep1_base[s][b]["phase_range"] for s in satellite_order]
        phi2 = [obs_ep2_rover[s][b]["phase_range"] for s in satellite_order]
        phi2 += [obs_ep2_base[s][b]["phase_range"] for s in satellite_order]

        dd_obs_ep1[b] = dd_matrix @ np.array(phi1, dtype=float)
        dd_obs_ep2[b] = dd_matrix @ np.array(phi2, dtype=float)

        cpb1 = [
            obs_ep1_rover[s][b]["code_range"] / wl - obs_ep1_rover[s][b]["phase_range"]
            for s in satellite_order
        ]
        cpb1 += [
            obs_ep1_base[s][b]["code_range"] / wl - obs_ep1_base[s][b]["phase_range"]
            for s in satellite_order
        ]
        cpb2 = [
            obs_ep2_rover[s][b]["code_range"] / wl - obs_ep2_rover[s][b]["phase_range"]
            for s in satellite_order
        ]
        cpb2 += [
            obs_ep2_base[s][b]["code_range"] / wl - obs_ep2_base[s][b]["phase_range"]
            for s in satellite_order
        ]

        dd_code_phase_bias_ep1[b] = dd_matrix @ np.array(cpb1, dtype=float)
        dd_code_phase_bias_ep2[b] = dd_matrix @ np.array(cpb2, dtype=float)

    num_bands = len(bands)
    m = len(satellite_order)
    d_left = dd_matrix[:, :m]

    n_obs = 2 * num_bands * num_dd
    w = np.zeros((n_obs, n_obs), dtype=float)
    c_dd_base = dd_matrix @ dd_matrix.T
    for ib, b in enumerate(bands):
        s2 = sigma_phi[b] ** 2
        c_block = s2 * c_dd_base
        w_block = np.linalg.inv(c_block)
        for ie in range(2):
            row = (2 * ib + ie) * num_dd
            w[row : row + num_dd, row : row + num_dd] = w_block

    dd_phase_biases: dict[str, np.ndarray] = {
        b: np.zeros(num_dd, dtype=float) for b in bands
    }

    for _ in range(10):
        n_unk = 3 + num_bands * num_dd
        dy = np.zeros(n_obs, dtype=float)
        h = np.zeros((n_obs, n_unk), dtype=float)

        los_ep1 = np.array(
            [
                (satpos_ep1_rover[s] - rover_pos)
                / np.linalg.norm(satpos_ep1_rover[s] - rover_pos)
                for s in satellite_order
            ],
            dtype=float,
        )
        los_ep2 = np.array(
            [
                (satpos_ep2_rover[s] - rover_pos)
                / np.linalg.norm(satpos_ep2_rover[s] - rover_pos)
                for s in satellite_order
            ],
            dtype=float,
        )

        for ib, b in enumerate(bands):
            wl = WLENS[b]
            row_ep1 = (2 * ib) * num_dd
            row_ep2 = (2 * ib + 1) * num_dd
            col_n = 3 + ib * num_dd

            dd_rho_ep1 = dd_geometric_ranges(
                satpos_ep1_rover,
                satpos_ep1_base,
                satellite_order,
                dd_matrix,
                rover_pos,
                base_pos,
            )
            dd_rho_ep2 = dd_geometric_ranges(
                satpos_ep2_rover,
                satpos_ep2_base,
                satellite_order,
                dd_matrix,
                rover_pos,
                base_pos,
            )

            dy[row_ep1 : row_ep1 + num_dd] = dd_obs_ep1[b] - (
                dd_rho_ep1 / wl + dd_phase_biases[b]
            )
            dy[row_ep2 : row_ep2 + num_dd] = dd_obs_ep2[b] - (
                dd_rho_ep2 / wl + dd_phase_biases[b]
            )

            h[row_ep1 : row_ep1 + num_dd, :3] = -d_left @ los_ep1 / wl
            h[row_ep2 : row_ep2 + num_dd, :3] = -d_left @ los_ep2 / wl
            h[row_ep1 : row_ep1 + num_dd, col_n : col_n + num_dd] = np.eye(num_dd)
            h[row_ep2 : row_ep2 + num_dd, col_n : col_n + num_dd] = np.eye(num_dd)

        htw = h.T @ w
        dx, *_ = np.linalg.lstsq(htw @ h, htw @ dy, rcond=None)

        rover_pos = rover_pos + dx[:3]
        for ib, b in enumerate(bands):
            dd_phase_biases[b] += dx[3 + ib * num_dd : 3 + (ib + 1) * num_dd]

        if np.linalg.norm(dx[:3]) < 1e-5 and np.linalg.norm(dx[3:]) < 1e-5:
            break

    dd_phase_biases_int = {
        b: np.round(v).astype(int) for b, v in dd_phase_biases.items()
    }

    def recalc_one_epoch(
        satpos_rover: dict[str, np.ndarray],
        satpos_base: dict[str, np.ndarray],
        dd_obs_by_band: dict[str, np.ndarray],
        init_pos: np.ndarray,
    ) -> np.ndarray:
        pos = init_pos.copy()
        n_obs_epoch = num_bands * num_dd

        w_epoch = np.zeros((n_obs_epoch, n_obs_epoch), dtype=float)
        for ib, b in enumerate(bands):
            s2 = sigma_phi[b] ** 2
            c_block = s2 * c_dd_base
            w_epoch[
                ib * num_dd : (ib + 1) * num_dd, ib * num_dd : (ib + 1) * num_dd
            ] = np.linalg.inv(c_block)

        for _ in range(10):
            dy_epoch = np.zeros(n_obs_epoch, dtype=float)
            h_epoch = np.zeros((n_obs_epoch, 3), dtype=float)

            los = np.array(
                [
                    (satpos_rover[s] - pos) / np.linalg.norm(satpos_rover[s] - pos)
                    for s in satellite_order
                ],
                dtype=float,
            )

            for ib, b in enumerate(bands):
                wl = WLENS[b]
                row = ib * num_dd
                dd_rho = dd_geometric_ranges(
                    satpos_rover, satpos_base, satellite_order, dd_matrix, pos, base_pos
                )
                dy_epoch[row : row + num_dd] = dd_obs_by_band[b] - (
                    dd_rho / wl + dd_phase_biases_int[b]
                )
                h_epoch[row : row + num_dd, :] = -d_left @ los / wl

            htw_epoch = h_epoch.T @ w_epoch
            dpos, *_ = np.linalg.lstsq(
                htw_epoch @ h_epoch, htw_epoch @ dy_epoch, rcond=None
            )
            pos = pos + dpos
            if np.linalg.norm(dpos) < 1e-5:
                break
        return pos

    rover_pos_ep1 = recalc_one_epoch(
        satpos_ep1_rover, satpos_ep1_base, dd_obs_ep1, rover_pos
    )
    rover_pos_ep2 = recalc_one_epoch(
        satpos_ep2_rover, satpos_ep2_base, dd_obs_ep2, rover_pos
    )

    result = {
        "base_db": str(base_db),
        "rover_db": str(rover_db),
        "config_json": str(config_path),
        "epochs": {
            "start_time": epoch1.isoformat(sep=" "),
            "end_time": epoch2.isoformat(sep=" "),
        },
        "bands": bands,
        "satellite_blocks": [list(block) for block in satellite_blocks],
        "satellite_order": satellite_order,
        "dd_matrix": dd_matrix.tolist(),
        "wavelengths": {b: float(WLENS[b]) for b in bands},
        "sigma_phi": sigma_phi,
        "dd_observations": {
            "epoch1": {b: dd_obs_ep1[b].tolist() for b in bands},
            "epoch2": {b: dd_obs_ep2[b].tolist() for b in bands},
        },
        "dd_code_phase_biases": {
            "epoch1": {b: dd_code_phase_bias_ep1[b].tolist() for b in bands},
            "epoch2": {b: dd_code_phase_bias_ep2[b].tolist() for b in bands},
        },
        "dd_phase_biases_float": {b: dd_phase_biases[b].tolist() for b in bands},
        "dd_phase_biases_int": {b: dd_phase_biases_int[b].tolist() for b in bands},
        "positions": {
            "base_position_ecef": base_pos.tolist(),
            "rover_position_float_solution": rover_pos.tolist(),
            "rover_position_epoch1_fixed": rover_pos_ep1.tolist(),
            "rover_position_epoch2_fixed": rover_pos_ep2.tolist(),
        },
        "baseline_lengths_m": {
            "float_solution": float(np.linalg.norm(rover_pos - base_pos)),
            "epoch1_fixed": float(np.linalg.norm(rover_pos_ep1 - base_pos)),
            "epoch2_fixed": float(np.linalg.norm(rover_pos_ep2 - base_pos)),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Bands: {bands}")
    print(f"Satellite blocks: {satellite_blocks}")
    print(f"Rover float solution: {rover_pos}")
    print(f"Rover epoch1 fixed : {rover_pos_ep1}")
    print(f"Rover epoch2 fixed : {rover_pos_ep2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
