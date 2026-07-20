from pathlib import Path
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from app.gnss.coordinates import ecef_to_enu_matrix
from app.gnss.relative_positioning import (
    RelativePositionEpochResult,
)


def plot_enu_time_series(
    epoch_results: list[RelativePositionEpochResult],
    output_path: Path,
) -> None:
    """Plot ENU time series (E/N/U in 3 rows) with base station as origin."""
    if not epoch_results:
        return

    base_llh = epoch_results[0].base_position_wgs84
    enu_rot = ecef_to_enu_matrix(base_llh.lat_deg, base_llh.lon_deg)
    base_ecef = np.array(epoch_results[0].base_position_ecef, dtype=float)

    times = [r.datetime for r in epoch_results]
    enu = []
    for r in epoch_results:
        rover_ecef = np.array(r.rover_position_ecef, dtype=float)
        diff = rover_ecef - base_ecef
        enu.append(enu_rot @ diff)
    enu_arr = np.array(enu)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["E", "N", "U"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i in range(3):
        axes[i].plot(times, enu_arr[:, i], color=colors[i], linewidth=1.3)
        axes[i].set_ylabel(f"{labels[i]} [m]")
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel("Time")
    fig.suptitle("Rover ENU Time Series (origin: Base)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


_WEBMERC_R = 6378137.0
_WEBMERC_ORIGIN_SHIFT = math.pi * _WEBMERC_R


def _lonlat_to_tile_xy(
    lon_deg: float, lat_deg: float, zoom: int
) -> tuple[float, float]:
    lat_deg = max(min(lat_deg, 85.05112878), -85.05112878)
    n = 2**zoom
    xt = (lon_deg + 180.0) / 360.0 * n
    lat_rad = math.radians(lat_deg)
    yt = (
        (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * n
    )
    return xt, yt


def _lonlat_to_webmerc(
    lon_deg: np.ndarray, lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lon_rad = np.radians(lon_deg)
    lat_clamped = np.clip(lat_deg, -85.05112878, 85.05112878)
    lat_rad = np.radians(lat_clamped)
    x = _WEBMERC_R * lon_rad
    y = _WEBMERC_R * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))
    return x, y


def _fetch_png_array(tile_url: str) -> np.ndarray:
    req = Request(tile_url, headers={"User-Agent": "gnssraw-rp-bias/1.0"})
    with urlopen(req, timeout=10) as resp:
        data = resp.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _build_tile_mosaic(
    url_template: str,
    zoom: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> np.ndarray:
    rows = []
    n = 2**zoom
    for y in range(y_min, y_max + 1):
        row_imgs = []
        for x in range(x_min, x_max + 1):
            x_wrapped = x % n
            y_clamped = min(max(y, 0), n - 1)
            url = url_template.format(z=zoom, x=x_wrapped, y=y_clamped)
            row_imgs.append(_fetch_png_array(url))
        rows.append(np.concatenate(row_imgs, axis=1))
    return np.concatenate(rows, axis=0)


def plot_track_on_background_maps(
    epoch_results: list[RelativePositionEpochResult],
    output_dir: Path,
    zoom: int = 17,
) -> list[Path]:
    """Plot rover 2D track on ESRI/OSM/GSI backgrounds and save images.

    Returns list of saved file paths. Background failures are skipped.
    """
    if not epoch_results:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    lats = np.array(
        [r.rover_position_wgs84.lat_deg for r in epoch_results], dtype=float
    )
    lons = np.array(
        [r.rover_position_wgs84.lon_deg for r in epoch_results], dtype=float
    )

    lon_pad = 0.001
    lat_pad = 0.001
    min_lon = float(np.min(lons) - lon_pad)
    max_lon = float(np.max(lons) + lon_pad)
    min_lat = float(np.min(lats) - lat_pad)
    max_lat = float(np.max(lats) + lat_pad)

    tx0, ty0 = _lonlat_to_tile_xy(min_lon, max_lat, zoom)
    tx1, ty1 = _lonlat_to_tile_xy(max_lon, min_lat, zoom)

    x_min = int(math.floor(min(tx0, tx1)))
    x_max = int(math.floor(max(tx0, tx1)))
    y_min = int(math.floor(min(ty0, ty1)))
    y_max = int(math.floor(max(ty0, ty1)))

    tile_margin = 1
    x_min -= tile_margin
    x_max += tile_margin
    y_min -= tile_margin
    y_max += tile_margin

    world_res = (2 * _WEBMERC_ORIGIN_SHIFT) / (256 * 2**zoom)
    extent_x_min = x_min * 256 * world_res - _WEBMERC_ORIGIN_SHIFT
    extent_x_max = (x_max + 1) * 256 * world_res - _WEBMERC_ORIGIN_SHIFT
    extent_y_max = _WEBMERC_ORIGIN_SHIFT - y_min * 256 * world_res
    extent_y_min = _WEBMERC_ORIGIN_SHIFT - (y_max + 1) * 256 * world_res

    track_x, track_y = _lonlat_to_webmerc(lons, lats)

    providers = {
        "esri_world_imagery": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "openstreetmap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "gsi_std": "https://cyberjapandata.gsi.go.jp/xyz/std/{z}/{x}/{y}.png",
    }

    saved_paths: list[Path] = []
    for name, template in providers.items():
        try:
            bg = _build_tile_mosaic(template, zoom, x_min, x_max, y_min, y_max)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            UnidentifiedImageError,
            ValueError,
        ) as exc:  # pragma: no cover - network dependent
            print(f"Skipping map plot for {name}: {exc}")
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            bg,
            extent=(extent_x_min, extent_x_max, extent_y_min, extent_y_max),
            origin="upper",
        )
        ax.plot(track_x, track_y, color="yellow", linewidth=2.0, label="Rover track")
        ax.scatter(track_x[0], track_y[0], c="lime", s=40, label="Start")
        ax.scatter(track_x[-1], track_y[-1], c="red", s=40, label="End")
        ax.set_title(f"Relative Positioning Track ({name})")
        ax.set_xlabel("Web Mercator X [m]")
        ax.set_ylabel("Web Mercator Y [m]")
        ax.legend(loc="best")
        ax.set_aspect("equal")

        out_path = output_dir / f"rp_bias_track_{name}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths
