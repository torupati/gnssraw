"""
Plot GNSS satellite ground tracks and current positions on a world map.

Uses skyfield for TLE-based orbit propagation and Cartopy for map projection.
TLE data is fetched automatically from CelesTrak (no account required).

Usage (constellation mode):
    uv run app/plot_orbit2d.py
    uv run app/plot_orbit2d.py -c GLONASS -p mercator -e 2027-06-01T12:00:00

Usage (single LEO satellite mode):
    uv run app/plot_orbit2d.py -s STARLINK-1007
    uv run app/plot_orbit2d.py -s 44713            # by NORAD catalog number
    uv run app/plot_orbit2d.py -s STARLINK-1007 -p mercator -e 2027-03-20T06:00:00

Supported constellations: GPS (default), GLONASS, Galileo, BeiDou, QZSS, IRNSS
Epoch format: ISO 8601, e.g. 2027-01-15T06:30:00  (interpreted as UTC)
"""

from __future__ import annotations

import argparse
import math
import urllib.parse
from datetime import datetime, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time

matplotlib.rcParams["figure.dpi"] = 120

MIN_LATITUDE = -80
MAX_LATITUDE = 80

# Supported map projections: display name -> Cartopy CRS factory
# Mercator is capped at ±85° latitude to avoid infinities at the poles.
PROJECTIONS: dict[str, tuple[str, object]] = {
    "robinson": ("Robinson", ccrs.Robinson()),
    "mercator": (
        "Mercator",
        ccrs.Mercator(min_latitude=MIN_LATITUDE, max_latitude=MAX_LATITUDE),
    ),
}

# CelesTrak TLE base URL (no auth required)
_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=TLE"
_CELESTRAK_NAME_URL = "https://celestrak.org/NORAD/elements/gp.php?NAME={name}&FORMAT=TLE"
_CELESTRAK_CATNR_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE"

# Default track duration for LEO satellites — overridden when --orbits is used
_LEO_TRACK_MINUTES = 95 * 2

# Constellation definitions:
#   display name -> (celestrak_group, orbital_period_minutes, name_filter_prefix | None)
# name_filter_prefix: if set, download the group then keep only satellites whose
#   name starts with one of the given prefixes (used when no dedicated group exists).
_ConstellationDef = tuple[str, int, list[str] | None]

CONSTELLATIONS: dict[str, _ConstellationDef] = {
    "GPS": ("gps-ops", 12 * 60, None),  # MEO ~12 h
    "GLONASS": ("glo-ops", 675, None),  # MEO ~11.25 h
    "Galileo": ("galileo", 12 * 60, None),  # MEO ~12 h
    "BeiDou": ("beidou", 12 * 60, None),  # MEO ~12 h (also GEO/IGSO)
    "QZSS": ("gnss", 24 * 60, ["QZS"]),  # GEO/GSO ~24 h; no dedicated group
    "IRNSS": ("gnss", 24 * 60, ["IRNSS", "NVS"]),  # GEO/GSO ~24 h; no dedicated group
}

# Case-insensitive lookup key -> canonical name
_CONSTELLATION_LOOKUP: dict[str, str] = {k.lower(): k for k in CONSTELLATIONS}

# Default plot epoch: 2027-01-15 00:00:00 UTC
EPOCH_UTC = datetime(2027, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

_EPOCH_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def parse_epoch(s: str) -> datetime:
    """Parse an ISO 8601 datetime string and return a UTC-aware datetime."""
    for fmt in _EPOCH_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse epoch '{s}'. Expected ISO 8601 format, e.g. 2027-01-15T06:30:00")


def resolve_constellation(name: str) -> str:
    """Return canonical constellation name, raising ValueError for unknown names."""
    key = name.strip().lower()
    if key not in _CONSTELLATION_LOOKUP:
        valid = ", ".join(CONSTELLATIONS)
        raise ValueError(f"Unknown constellation '{name}'. Valid options: {valid}")
    return _CONSTELLATION_LOOKUP[key]


def fetch_satellites(constellation: str) -> tuple[list[EarthSatellite], object]:
    """Download TLE data for the given constellation from CelesTrak."""
    group, _, name_prefixes = CONSTELLATIONS[constellation]
    url = _CELESTRAK_BASE.format(group=group)
    ts = load.timescale()
    # Use a constellation-specific filename to avoid skyfield reusing the same
    # cache file (gp.php) for all constellations.
    cache_filename = f"tle_{constellation.lower()}.txt"
    satellites = load.tle_file(url, filename=cache_filename)
    # Filter by name prefix when the group contains multiple constellations (e.g. gnss)
    if name_prefixes is not None:
        satellites = [s for s in satellites if any(s.name.startswith(prefix) for prefix in name_prefixes)]
    print(f"Loaded {len(satellites)} {constellation} satellites from CelesTrak")
    return satellites, ts


def fetch_satellite_by_spec(spec: str, ts) -> EarthSatellite:
    """Fetch a single satellite TLE from CelesTrak by name or NORAD catalog number."""
    spec = spec.strip()
    if spec.isdigit():
        url = _CELESTRAK_CATNR_URL.format(catnr=spec)
        cache_filename = f"tle_catnr_{spec}.txt"
    else:
        url = _CELESTRAK_NAME_URL.format(name=urllib.parse.quote(spec))
        safe = spec.replace(" ", "_").replace("/", "_")
        cache_filename = f"tle_name_{safe}.txt"

    satellites: list[EarthSatellite] = []
    try:
        satellites = load.tle_file(url, filename=cache_filename)
    except OSError as exc:
        raise ValueError(f"Satellite '{spec}' not found on CelesTrak (HTTP 404). It may have de-orbited. Try a different name or NORAD ID.") from exc
    if not satellites:
        raise ValueError(f"No satellite found for '{spec}'.")
    if len(satellites) > 1:
        names = ", ".join(s.name for s in satellites[:5])
        suffix = "..." if len(satellites) > 5 else ""
        print(f"  {len(satellites)} matches: {names}{suffix}")
        print(f"  Using: {satellites[0].name}")
    else:
        print(f"  Found: {satellites[0].name}")
    return satellites[0]


def orbital_period_minutes(sat: EarthSatellite) -> float:
    """Compute orbital period in minutes from TLE mean motion."""
    return 2.0 * math.pi / sat.model.no_kozai


def build_epoch_time(ts, epoch_utc: datetime) -> Time:
    """Convert a UTC datetime to a skyfield Time."""
    return ts.from_datetime(epoch_utc)


def compute_ground_track(
    sat: EarthSatellite,
    ts,
    t_epoch: Time,
    duration_minutes: int = 12 * 60,
    steps: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Compute the ground track (lat/lon) for one orbital period centred on t_epoch.

    Args:
        steps: Number of sample points. Defaults to max(360, duration_minutes)
               so there is at least one sample per minute.

    Returns:
        lons: list of longitudes in degrees
        lats: list of latitudes in degrees
    """
    if steps is None:
        steps = max(360, duration_minutes)
    half = duration_minutes / 2.0
    t_start = ts.tt_jd(t_epoch.tt - half / 1440.0)
    t_end = ts.tt_jd(t_epoch.tt + half / 1440.0)
    times = ts.tt_jd([t_start.tt + i * (t_end.tt - t_start.tt) / (steps - 1) for i in range(steps)])
    subpoints = wgs84.subpoint_of(sat.at(times))
    return subpoints.longitude.degrees.tolist(), subpoints.latitude.degrees.tolist()


def split_antimeridian(lons: list[float], lats: list[float]) -> list[tuple]:
    """
    Split a ground track at the antimeridian (±180°) so Cartopy does not
    draw long horizontal lines across the map.

    Returns a list of (lons_segment, lats_segment) tuples.
    """
    segments: list[tuple] = []
    seg_lons: list[float] = [lons[0]]
    seg_lats: list[float] = [lats[0]]

    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180.0:
            segments.append((seg_lons, seg_lats))
            seg_lons = []
            seg_lats = []
        seg_lons.append(lons[i])
        seg_lats.append(lats[i])

    if seg_lons:
        segments.append((seg_lons, seg_lats))
    return segments


def _draw_satellite_tracks(
    ax,
    satellites: list[EarthSatellite],
    ts,
    t_epoch: Time,
    duration_minutes: int,
    single_color: str | None = None,
) -> None:
    """
    Draw ground tracks and current-position markers for each satellite.

    Args:
        ax: Cartopy GeoAxes
        satellites: List of EarthSatellite objects
        ts: Skyfield timescale (unused here, kept for symmetry)
        t_epoch: Skyfield Time for the plot epoch
        duration_minutes: Length of ground track to draw
        single_color: If given, use this color for all satellites (LEO mode).
                      If None, cycle through tab20 colormap.
    """
    cmap = plt.get_cmap("tab20")
    num_sats = len(satellites)

    for idx, sat in enumerate(satellites):
        color = single_color if single_color else cmap(idx / max(num_sats - 1, 1))

        try:
            lons, lats = compute_ground_track(sat, ts, t_epoch, duration_minutes=duration_minutes)
        except Exception as exc:
            print(f"  Warning: {sat.name} ground track failed: {exc}")
            continue

        for seg_lons, seg_lats in split_antimeridian(lons, lats):
            if len(seg_lons) < 2:
                continue
            ax.plot(
                seg_lons,
                seg_lats,
                transform=ccrs.Geodetic(),
                color=color,
                linewidth=0.7,
                alpha=0.6,
                zorder=2,
            )

        try:
            subpoint = wgs84.subpoint_of(sat.at(t_epoch))
            lon0 = float(subpoint.longitude.degrees)
            lat0 = float(subpoint.latitude.degrees)
            ax.plot(
                lon0,
                lat0,
                transform=ccrs.Geodetic(),
                marker="o",
                color=color,
                markersize=5 if single_color is None else 8,
                markeredgecolor="black",
                markeredgewidth=0.4,
                zorder=3,
            )
            ax.text(
                lon0,
                lat0 + 2.5,
                sat.name,
                transform=ccrs.Geodetic(),
                fontsize=4.5,
                color=color,
                ha="center",
                va="bottom",
                zorder=4,
            )
        except Exception as exc:
            print(f"  Warning: {sat.name} position failed: {exc}")


def _make_map_axes(projection: str) -> tuple:
    """Create figure and Cartopy axes with standard map background."""
    proj_label, crs = PROJECTIONS[projection]
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(1, 1, 1, projection=crs)
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#e8e4d5", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#c6ddf0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555555", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#888888", zorder=1)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#aaaaaa", linestyle="--", zorder=1)
    return fig, ax, proj_label


def plot_orbits(
    constellation: str = "GPS",
    projection: str = "robinson",
    epoch_utc: datetime = EPOCH_UTC,
    output_path: str | None = None,
) -> None:
    """Plot all satellites in a GNSS constellation."""
    _, duration_minutes, _ = CONSTELLATIONS[constellation]
    satellites, ts = fetch_satellites(constellation)
    t_epoch = build_epoch_time(ts, epoch_utc)

    fig, ax, proj_label = _make_map_axes(projection)
    _draw_satellite_tracks(ax, satellites, ts, t_epoch, duration_minutes)

    epoch_str = epoch_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    ax.set_title(
        f"{constellation} Satellite Ground Tracks & Positions  [{proj_label} projection]\n{epoch_str}",
        fontsize=13,
        pad=10,
    )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_single_orbit(
    sat: EarthSatellite,
    ts,
    projection: str = "robinson",
    epoch_utc: datetime = EPOCH_UTC,
    num_orbits: int = 2,
    output_path: str | None = None,
) -> None:
    """Plot the ground track and current position of a single satellite."""
    period_min = orbital_period_minutes(sat)
    duration_minutes = int(period_min * num_orbits)
    t_epoch = build_epoch_time(ts, epoch_utc)
    fig, ax, proj_label = _make_map_axes(projection)
    _draw_satellite_tracks(ax, [sat], ts, t_epoch, duration_minutes, single_color="#e84545")
    epoch_str = epoch_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    orbit_label = f"{num_orbits} orbit{'s' if num_orbits != 1 else ''}"
    ax.set_title(
        f"{sat.name}  ({orbit_label}, T={period_min:.1f} min)  [{proj_label} projection]\n{epoch_str}",
        fontsize=13,
        pad=10,
    )
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main() -> None:
    valid_names = ", ".join(CONSTELLATIONS)
    valid_projections = ", ".join(PROJECTIONS)
    parser = argparse.ArgumentParser(
        description="Plot GNSS constellation or single LEO satellite ground tracks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported constellations: {valid_names}\n"
            f"Supported projections:    {valid_projections}\n"
            f"Epoch format:             ISO 8601, e.g. 2027-06-01T12:00:00 (UTC)\n"
            f"\n"
            f"Single satellite examples:\n"
            f"  -s STARLINK-1007           (name match, case-insensitive)\n"
            f"  -s 44713                   (NORAD catalog number)"
        ),
    )
    parser.add_argument(
        "-c",
        "--constellation",
        metavar="NAME",
        default="GPS",
        help=f"GNSS constellation (default: GPS). One of: {valid_names}. Ignored when -s is given.",
    )
    parser.add_argument(
        "-s",
        "--satellite",
        metavar="NAME_OR_ID",
        default=None,
        help=("Plot a single LEO satellite by name (e.g. STARLINK-1007) or NORAD catalog number (e.g. 44713). Overrides --constellation."),
    )
    parser.add_argument(
        "-p",
        "--projection",
        metavar="PROJ",
        default="robinson",
        choices=list(PROJECTIONS),
        help="Map projection (default: robinson). One of: %(choices)s",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        metavar="DATETIME",
        default=None,
        help=("Plot epoch in ISO 8601 UTC (default: 2027-01-15T00:00:00). Example: 2027-06-01T12:30:00"),
    )
    parser.add_argument(
        "-n",
        "--orbits",
        metavar="N",
        type=int,
        default=2,
        help="Number of orbital periods to show for a single satellite (default: 2). Only used with -s.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="Save plot to file (e.g. orbits.png) instead of displaying it.",
    )
    args = parser.parse_args()

    epoch_utc = EPOCH_UTC
    if args.epoch is not None:
        try:
            epoch_utc = parse_epoch(args.epoch)
        except ValueError as exc:
            parser.error(str(exc))

    if args.satellite is not None:
        ts = load.timescale()
        try:
            sat = fetch_satellite_by_spec(args.satellite, ts)
        except ValueError as exc:
            parser.error(str(exc))
        plot_single_orbit(
            sat=sat,
            ts=ts,
            projection=args.projection,
            epoch_utc=epoch_utc,
            num_orbits=args.orbits,
            output_path=args.output,
        )
    else:
        try:
            constellation = resolve_constellation(args.constellation)
        except ValueError as exc:
            parser.error(str(exc))
        plot_orbits(
            constellation=constellation,
            projection=args.projection,
            epoch_utc=epoch_utc,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
