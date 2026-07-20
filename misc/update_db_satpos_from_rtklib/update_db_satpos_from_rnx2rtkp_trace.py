"""Rewrite satellite positions in a GNSS SQLite DB from RTKLIB rnx2rtkp trace output.

The script parses `rnx2rtkp.trace` lines like:
    4 2026/06/13 09:59:59.922716 sat=10 rs=  4908090.034  13755005.467  22449934.113 ...

and writes the `rs=` ECEF coordinates into the matching epoch/satellite rows in the
SQLite database produced by app/spp.py.

By default this runs in dry-run mode. Use `--apply` to actually update the DB.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import shutil
from pathlib import Path
from typing import TypedDict

import numpy as np

from app.gnss.database import GnssDatabase, Epoch, Satellite, SatellitePosition

POS_RE = re.compile(
    r"^\d+\s+(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+sat=(\d+)\s+rs="
    r"\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+dts=",
    re.MULTILINE,
)


class TracePosition(TypedDict):
    datetime: dt.datetime
    nano_second: int
    x: float
    y: float
    z: float


class DiffRow(TypedDict):
    epoch_datetime: str
    satellite_id: str
    status: str
    before_x_m: float | None
    before_y_m: float | None
    before_z_m: float | None
    after_x_m: float
    after_y_m: float
    after_z_m: float
    delta_x_m: float | None
    delta_y_m: float | None
    delta_z_m: float | None
    delta_norm_m: float | None


def trace_position_time_to_epoch(value: str) -> dt.datetime:
    """Convert trace rs-line timestamp to observation epoch.

    RTKLIB trace `rs=` lines are stamped a fraction of a second before the epoch
    (e.g. 09:59:59.922 for the 10:00:00 epoch, 10:00:29.928 for 10:00:30).
    """
    pos_time = dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f")
    return (pos_time + dt.timedelta(milliseconds=100)).replace(microsecond=0)


def rtklib_satno_to_id(sat_no: int) -> str:
    """Convert RTKLIB internal satellite number to RINEX-like satellite ID."""
    if 1 <= sat_no <= 32:
        return f"G{sat_no:02d}"
    if 33 <= sat_no <= 59:
        return f"R{sat_no - 32:02d}"
    if 60 <= sat_no <= 95:
        return f"E{sat_no - 59:02d}"
    if 96 <= sat_no <= 105:
        return f"J{sat_no - 95:02d}"
    if 106 <= sat_no <= 168:
        return f"C{sat_no - 105:02d}"
    if 169 <= sat_no <= 182:
        return f"I{sat_no - 168:02d}"
    if 193 <= sat_no <= 202:
        return f"S{sat_no - 192:02d}"
    raise ValueError(f"Unsupported RTKLIB satellite number: {sat_no}")


def parse_trace_positions(
    trace_path: Path,
    include_zero: bool = False,
) -> dict[dt.datetime, dict[str, TracePosition]]:
    """Parse trace into epoch -> satellite_id -> position dict."""
    positions: dict[dt.datetime, dict[str, TracePosition]] = {}

    with trace_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_pos = POS_RE.search(line)
            if not m_pos:
                continue

            current_epoch = trace_position_time_to_epoch(m_pos.group(1))
            positions.setdefault(current_epoch, {})
            sat_no = int(m_pos.group(2))
            sat_id = rtklib_satno_to_id(sat_no)
            x = float(m_pos.group(3))
            y = float(m_pos.group(4))
            z = float(m_pos.group(5))

            if not include_zero and np.allclose([x, y, z], 0.0):
                continue

            positions[current_epoch][sat_id] = {
                "datetime": current_epoch,
                "nano_second": current_epoch.microsecond * 1000,
                "x": x,
                "y": y,
                "z": z,
            }

    return positions


def create_backup(db_path: Path) -> Path:
    """Create timestamped backup next to the target DB."""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}.backup_{timestamp}{db_path.suffix}")
    shutil.copy2(db_path, backup_path)
    return backup_path


def write_diff_csv(diff_rows: list[DiffRow], csv_path: Path) -> None:
    """Write position before/after differences to CSV."""
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch_datetime",
                "satellite_id",
                "status",
                "before_x_m",
                "before_y_m",
                "before_z_m",
                "after_x_m",
                "after_y_m",
                "after_z_m",
                "delta_x_m",
                "delta_y_m",
                "delta_z_m",
                "delta_norm_m",
            ]
        )
        for row in diff_rows:
            writer.writerow(
                [
                    row["epoch_datetime"],
                    row["satellite_id"],
                    row["status"],
                    row["before_x_m"],
                    row["before_y_m"],
                    row["before_z_m"],
                    row["after_x_m"],
                    row["after_y_m"],
                    row["after_z_m"],
                    row["delta_x_m"],
                    row["delta_y_m"],
                    row["delta_z_m"],
                    row["delta_norm_m"],
                ]
            )


def update_database(
    db_path: Path,
    positions_by_epoch: dict[dt.datetime, dict[str, TracePosition]],
    apply: bool,
) -> tuple[dict[str, int], list[DiffRow]]:
    """Update satellite positions in DB, preserving existing clock_bias."""
    db = GnssDatabase(db_path)
    session = db.Session()
    diff_rows: list[DiffRow] = []

    stats = {
        "epochs_in_trace": len(positions_by_epoch),
        "epochs_found_in_db": 0,
        "epochs_missing_in_db": 0,
        "sat_positions_in_trace": 0,
        "sat_rows_found": 0,
        "sat_rows_missing": 0,
        "positions_updated": 0,
        "positions_created": 0,
    }

    try:
        for epoch_dt, sat_positions in sorted(positions_by_epoch.items()):
            stats["sat_positions_in_trace"] += len(sat_positions)
            epoch_row = session.query(Epoch).filter_by(datetime=epoch_dt).one_or_none()
            if epoch_row is None:
                stats["epochs_missing_in_db"] += 1
                continue

            stats["epochs_found_in_db"] += 1
            sat_map = {
                sat.satellite_id: sat
                for sat in session.query(Satellite).filter_by(epoch_id=epoch_row.id).all()
            }

            for sat_id, pos_data in sat_positions.items():
                sat_row = sat_map.get(sat_id)
                if sat_row is None:
                    stats["sat_rows_missing"] += 1
                    continue

                stats["sat_rows_found"] += 1
                sat_pos = (
                    session.query(SatellitePosition)
                    .filter_by(satellite_id=sat_row.id)
                    .one_or_none()
                )
                after_x = float(pos_data["x"])
                after_y = float(pos_data["y"])
                after_z = float(pos_data["z"])
                before_x: float | None
                before_y: float | None
                before_z: float | None
                if sat_pos is None:
                    before_x = None
                    before_y = None
                    before_z = None
                    sat_pos = SatellitePosition(
                        satellite_id=sat_row.id,
                        datetime=epoch_dt,
                        nano_second=int(pos_data["nano_second"]),
                        x=after_x,
                        y=after_y,
                        z=after_z,
                        clock_bias=None,
                    )
                    session.add(sat_pos)
                    stats["positions_created"] += 1
                    status = "created"
                else:
                    before_x = float(sat_pos.x)
                    before_y = float(sat_pos.y)
                    before_z = float(sat_pos.z)
                    sat_pos.datetime = epoch_dt
                    sat_pos.nano_second = int(pos_data["nano_second"])
                    sat_pos.x = after_x
                    sat_pos.y = after_y
                    sat_pos.z = after_z
                    stats["positions_updated"] += 1
                    status = "updated"

                delta_x: float | None
                delta_y: float | None
                delta_z: float | None
                delta_norm: float | None
                if before_x is None or before_y is None or before_z is None:
                    delta_x = None
                    delta_y = None
                    delta_z = None
                    delta_norm = None
                else:
                    delta_x = after_x - before_x
                    delta_y = after_y - before_y
                    delta_z = after_z - before_z
                    delta_norm = float(np.linalg.norm([delta_x, delta_y, delta_z]))

                diff_rows.append(
                    {
                        "epoch_datetime": epoch_dt.isoformat(sep=" "),
                        "satellite_id": sat_id,
                        "status": status,
                        "before_x_m": before_x,
                        "before_y_m": before_y,
                        "before_z_m": before_z,
                        "after_x_m": after_x,
                        "after_y_m": after_y,
                        "after_z_m": after_z,
                        "delta_x_m": delta_x,
                        "delta_y_m": delta_y,
                        "delta_z_m": delta_z,
                        "delta_norm_m": delta_norm,
                    }
                )

        if apply:
            session.commit()
        else:
            session.rollback()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return stats, diff_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rewrite satellite positions in a GNSS DB from rnx2rtkp.trace"
    )
    parser.add_argument("--db", required=True, help="Target SQLite DB path")
    parser.add_argument("--trace", required=True, help="rnx2rtkp.trace path")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply changes to DB (default: dry-run)",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Also write all-zero rs positions (default: skip zeros)",
    )
    parser.add_argument(
        "--diff-csv",
        default=None,
        help="CSV path for before/after/delta satellite position differences",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    trace_path = Path(args.trace)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    diff_csv_path = (
        Path(args.diff_csv)
        if args.diff_csv is not None
        else db_path.with_name(f"{db_path.stem}_satpos_diff.csv")
    )

    backup_path = None
    if args.apply:
        backup_path = create_backup(db_path)

    positions_by_epoch = parse_trace_positions(trace_path, include_zero=args.include_zero)
    stats, diff_rows = update_database(db_path, positions_by_epoch, apply=args.apply)
    write_diff_csv(diff_rows, diff_csv_path)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"diff_csv: {diff_csv_path}")
    if backup_path is not None:
        print(f"backup_db: {backup_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
