"""Data structures and I/O helpers for relative positioning results."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Wgs84Position:
    """Geodetic coordinates in WGS84."""

    lat_deg: float
    lon_deg: float
    height_m: float


@dataclass
class RelativePositionEpochResult:
    """Relative positioning result for one epoch."""

    datetime: dt.datetime
    rover_position_ecef: list[float]
    rover_position_wgs84: Wgs84Position
    base_position_ecef: list[float]
    base_position_wgs84: Wgs84Position
    num_satellites: int
    baseline_length: float
    iterations: int
    residual_norm: float

    def to_json_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "datetime": self.datetime.isoformat(sep=" "),
            "position": self.rover_position_ecef,
            "position_wgs84": {
                "lat_deg": self.rover_position_wgs84.lat_deg,
                "lon_deg": self.rover_position_wgs84.lon_deg,
                "height_m": self.rover_position_wgs84.height_m,
            },
            "base_position": self.base_position_ecef,
            "base_position_wgs84": {
                "lat_deg": self.base_position_wgs84.lat_deg,
                "lon_deg": self.base_position_wgs84.lon_deg,
                "height_m": self.base_position_wgs84.height_m,
            },
            "num_satellites": self.num_satellites,
            "baseline_length": self.baseline_length,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
        }


def write_relative_position_results_csv(results: list[RelativePositionEpochResult], output_path: Path) -> None:
    """Write relative positioning results to CSV.

    Latitude/longitude are formatted with 9 decimal places, and heights with 4.
    """
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "datetime",
                "rover_lat_deg",
                "rover_lon_deg",
                "rover_height_m",
                "base_lat_deg",
                "base_lon_deg",
                "base_height_m",
                "num_satellites",
            ]
        )

        for row in results:
            writer.writerow(
                [
                    row.datetime.isoformat(sep=" "),
                    f"{row.rover_position_wgs84.lat_deg:.9f}",
                    f"{row.rover_position_wgs84.lon_deg:.9f}",
                    f"{row.rover_position_wgs84.height_m:.4f}",
                    f"{row.base_position_wgs84.lat_deg:.9f}",
                    f"{row.base_position_wgs84.lon_deg:.9f}",
                    f"{row.base_position_wgs84.height_m:.4f}",
                    row.num_satellites,
                ]
            )
