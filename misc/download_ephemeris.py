"""
Download SP3/CLK precise ephemeris for a RINEX observation file,
then compute nearest-SP3-epoch satellite positions for the first
observation epoch and print them to stdout.

Usage:
    uv run python misc/download_ephemeris.py [obs_file]
    uv run python misc/download_ephemeris.py dataset/0840164k.26o
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import gnss_lib_py as glp

from app.gnss.constants import CLIGHT
from app.gnss.satellite_signals import parse_rinex_observation_file

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_OBS_FILE        = _REPO / "dataset" / "0840164k.26o"
DEFAULT_SIGNAL_CODE_MAP = _REPO / "app" / ".signal_code_map.json"
SP3_DIR = _REPO / "data" / "ephemeris" / "sp3"
CLK_DIR = _REPO / "data" / "ephemeris" / "clk"


def _parse_rinex_header_time(line: str) -> datetime:
    fields = line[:60].split()
    if len(fields) < 6:
        raise ValueError(f"Unexpected RINEX time line: {line.rstrip()!r}")
    year, month, day, hour, minute = map(int, fields[:5])
    sec = float(fields[5])
    ws = int(sec)
    us = int(round((sec - ws) * 1_000_000))
    return datetime(year, month, day, hour, minute, ws, us, tzinfo=timezone.utc)


def read_observation_time_range(obs_file: Path) -> tuple[datetime, datetime]:
    first_obs: datetime | None = None
    last_obs: datetime | None = None
    with obs_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            if "END OF HEADER" in line:
                break
            if "TIME OF FIRST OBS" in line:
                first_obs = _parse_rinex_header_time(line)
            elif "TIME OF LAST OBS" in line:
                last_obs = _parse_rinex_header_time(line)
    if first_obs is None or last_obs is None:
        raise ValueError(f"TIME OF FIRST/LAST OBS not found in {obs_file}")
    return first_obs, last_obs


def _expected_sp3_path(obs_date: datetime) -> Path:
    doy = obs_date.timetuple().tm_yday
    return SP3_DIR / f"COD0MGXFIN_{obs_date.year}{doy:03d}0000_01D_05M_ORB.SP3"


def _expected_clk_path(obs_date: datetime) -> Path:
    doy = obs_date.timetuple().tm_yday
    return CLK_DIR / f"COD0MGXFIN_{obs_date.year}{doy:03d}0000_01D_30S_CLK.CLK"


def ensure_precise_ephemeris(
    first_obs: datetime,
    buffer_seconds: int = 3600,
) -> tuple[Path, Path]:
    """Return (sp3_path, clk_path), downloading only if not already cached."""
    sp3_path = _expected_sp3_path(first_obs)
    clk_path = _expected_clk_path(first_obs)

    if sp3_path.exists() and clk_path.exists():
        print(f"[cache] SP3: {sp3_path}")
        print(f"[cache] CLK: {clk_path}")
        return sp3_path, clk_path

    start = first_obs - timedelta(seconds=buffer_seconds)
    end   = first_obs + timedelta(seconds=buffer_seconds)
    gps_millis = np.array([
        float(np.atleast_1d(glp.datetime_to_gps_millis(np.array([start])))[0]),
        float(np.atleast_1d(glp.datetime_to_gps_millis(np.array([end])))[0]),
    ])

    if not sp3_path.exists():
        paths = glp.load_ephemeris(file_type="sp3", gps_millis=gps_millis, verbose=True)
        if paths:
            sp3_path = Path(paths[0])
    else:
        print(f"[cache] SP3: {sp3_path}")

    if not clk_path.exists():
        paths = glp.load_ephemeris(file_type="clk", gps_millis=gps_millis, verbose=True)
        if paths:
            clk_path = Path(paths[0])
    else:
        print(f"[cache] CLK: {clk_path}")

    return sp3_path, clk_path


class _Sp3Epoch:
    __slots__ = ("epoch", "pos")

    def __init__(
        self,
        epoch: datetime,
        pos: dict[str, tuple[float, float, float, float]],
    ) -> None:
        self.epoch = epoch
        self.pos = pos


def parse_sp3(sp3_file: Path) -> list[_Sp3Epoch]:
    """Parse SP3c/d. Positions km->m, clock us->s, sentinel->NaN."""
    records: list[_Sp3Epoch] = []
    cur_epoch: datetime | None = None
    cur_pos: dict[str, tuple[float, float, float, float]] = {}

    with sp3_file.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("*"):
                if cur_epoch is not None:
                    records.append(_Sp3Epoch(cur_epoch, cur_pos))
                parts = line[1:].split()
                y, mo, d, h, mi = map(int, parts[:5])
                sec = float(parts[5])
                ws  = int(sec)
                us  = int(round((sec - ws) * 1_000_000))
                cur_epoch = datetime(y, mo, d, h, mi, ws, us)
                cur_pos = {}
            elif line.startswith("P"):
                sat_id = line[1:4].strip()
                parts  = line[4:].split()
                if len(parts) < 4:
                    continue
                x_m    = float(parts[0]) * 1_000.0
                y_m    = float(parts[1]) * 1_000.0
                z_m    = float(parts[2]) * 1_000.0
                clk_us = float(parts[3])
                clk_s  = float("nan") if abs(clk_us) > 999_990.0 else clk_us * 1e-6
                cur_pos[sat_id] = (x_m, y_m, z_m, clk_s)

    if cur_epoch is not None:
        records.append(_Sp3Epoch(cur_epoch, cur_pos))

    return records


def nearest_sp3_state(
    records: list[_Sp3Epoch],
    sat_id: str,
    query_time: datetime,
) -> tuple[np.ndarray, float, float] | None:
    """(pos_ecef_m[3], clk_s, dt_s) from nearest SP3 epoch, or None."""
    best: _Sp3Epoch | None = None
    best_abs: float = float("inf")

    for rec in records:
        if sat_id not in rec.pos:
            continue
        diff = abs((rec.epoch - query_time).total_seconds())
        if diff < best_abs:
            best_abs = diff
            best = rec

    if best is None:
        return None

    x, y, z, clk_s = best.pos[sat_id]
    dt_s = (best.epoch - query_time).total_seconds()
    return np.array([x, y, z], dtype=float), clk_s, dt_s


def compute_and_print_first_epoch(
    obs_file: Path,
    sp3_path: Path,
    signal_code_map_path: Path,
) -> None:
    print(
        "\n[INFO] Computing nearest-SP3-epoch satellite positions for the first epoch:"
        f"\n       Observation file: {obs_file}"
        f"\n       SP3 file:         {sp3_path}"
        f"\n       Signal code map:  {signal_code_map_path}"
    )
    print(f"[INFO] Parsing signal code map: {signal_code_map_path}")
    with signal_code_map_path.open("r", encoding="utf-8") as fh:
        signal_code_map = json.load(fh)
    print(f"[INFO] Parsing RINEX observation file: {obs_file}")
    epochs = parse_rinex_observation_file(str(obs_file), signal_code_map)
    if not epochs:
        raise ValueError(f"No epochs found in {obs_file}")

    epoch       = epochs[0]
    print(f"\n[INFO] Parsing SP3 file: {sp3_path}")
    sp3_records = parse_sp3(sp3_path)

    print(f"\nEpoch : {epoch.datetime.isoformat()}")
    print(f"SP3   : {sp3_path.name}")
    hdr = (
        f"{'Sat':<6}"
        f" {'X [m]':>18}"
        f" {'Y [m]':>18}"
        f" {'Z [m]':>18}"
        f" {'Clk [s]':>15}"
        f" {'dt_SP3 [s]':>12}"
        f" {'PR [m]':>14}"
    )
    print(hdr)
    print("-" * len(hdr))

    for sat_id, sat_obs in epoch.iter_satellites():
        if "L1" not in sat_obs.signals:
            continue
        pr = sat_obs.signals["L1"].pseudorange
        if not np.isfinite(pr):
            continue

        tx_dt = epoch.datetime - timedelta(seconds=float(pr) / CLIGHT)
        state = nearest_sp3_state(sp3_records, sat_id, tx_dt)
        if state is None:
            print(f"{sat_id:<6} (no SP3 entry)")
            continue

        pos, clk_s, dt_s = state
        clk_str = f"{clk_s:15.9f}" if np.isfinite(clk_s) else f"{'NaN':>15}"
        print(
            f"{sat_id:<6}"
            f" {pos[0]:>18.3f}"
            f" {pos[1]:>18.3f}"
            f" {pos[2]:>18.3f}"
            f" {clk_str}"
            f" {dt_s:>12.1f}"
            f" {pr:>14.3f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download SP3/CLK precise ephemeris for a RINEX observation file "
            "and print nearest-SP3-epoch satellite positions for the first epoch."
        )
    )
    parser.add_argument(
        "obs_file",
        nargs="?",
        default=str(DEFAULT_OBS_FILE),
        help="RINEX observation file  (default: dataset/0840164k.26o)",
    )
    parser.add_argument(
        "--signal-code-map",
        default=str(DEFAULT_SIGNAL_CODE_MAP),
        help="Path to signal_code_map JSON",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=int,
        default=3600,
        help="Download margin around the observation span [s]  (default: 3600)",
    )

    args = parser.parse_args()
    obs_file = Path(args.obs_file)
    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")

    signal_code_map_path = Path(args.signal_code_map)
    if not signal_code_map_path.exists():
        raise FileNotFoundError(f"Signal code map not found: {signal_code_map_path}")

    first_obs, _ = read_observation_time_range(obs_file)
    sp3_path, _clk_path = ensure_precise_ephemeris(
        first_obs, buffer_seconds=args.buffer_seconds
    )

    print(
        "\n[INFO] Computing nearest-SP3-epoch satellite positions for the first epoch:"
        f"\n       Observation file: {obs_file}"
        f"\n       SP3 file:         {sp3_path}"
        f"\n       Signal code map:  {signal_code_map_path}"
        f"\n       Buffer seconds:   {args.buffer_seconds}"
    )
    compute_and_print_first_epoch(obs_file, sp3_path, signal_code_map_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
