"""
3D satellite orbit visualization using plotly.

Positions are propagated with skyfield (SGP4/TLE) in the GCRS inertial frame,
so each satellite's orbit appears as a fixed ellipse around the Earth.

Usage (constellation mode):
    uv run app/plot_orbit3d.py
    uv run app/plot_orbit3d.py -c GLONASS
    uv run app/plot_orbit3d.py -c GPS -o gps_3d.html

Usage (single satellite mode):
    uv run app/plot_orbit3d.py -s STARLINK-1008
    uv run app/plot_orbit3d.py -s 44714 -n 3 -e 2027-06-01T00:00:00

Supported constellations: GPS (default), GLONASS, Galileo, BeiDou, QZSS, IRNSS
Output: interactive HTML (default) or .html/.png file via -o
"""

from __future__ import annotations

import argparse
import math
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
from skyfield.api import load, EarthSatellite
from skyfield.timelib import Time

# --------------------------------------------------------------------------- #
# Constants & constellation definitions (mirrors plot_orbit2d.py)
# --------------------------------------------------------------------------- #

_RE_KM = 6371.0  # mean Earth radius

_CELESTRAK_BASE = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=TLE"
_CELESTRAK_NAME_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?NAME={name}&FORMAT=TLE"
)
_CELESTRAK_CATNR_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?CATNR={catnr}&FORMAT=TLE"
)

_ConstellationDef = tuple[str, int, list[str] | None]

CONSTELLATIONS: dict[str, _ConstellationDef] = {
    "GPS": ("gps-ops", 12 * 60, None),
    "GLONASS": ("glo-ops", 675, None),
    "Galileo": ("galileo", 12 * 60, None),
    "BeiDou": ("beidou", 12 * 60, None),
    "QZSS": ("gnss", 24 * 60, ["QZS"]),
    "IRNSS": ("gnss", 24 * 60, ["IRNSS", "NVS"]),
}

_CONSTELLATION_LOOKUP: dict[str, str] = {k.lower(): k for k in CONSTELLATIONS}

EPOCH_UTC = datetime(2027, 1, 15, 0, 0, 0, tzinfo=timezone.utc)

_EPOCH_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]

# Distinct colors for up to 32 satellites
_SAT_COLORS = [
    "#ef5350",
    "#42a5f5",
    "#66bb6a",
    "#ffa726",
    "#ab47bc",
    "#26c6da",
    "#d4e157",
    "#ec407a",
    "#8d6e63",
    "#78909c",
    "#5c6bc0",
    "#26a69a",
    "#ffee58",
    "#ff7043",
    "#29b6f6",
    "#9ccc65",
    "#ffa000",
    "#7e57c2",
    "#f06292",
    "#a5d6a7",
    "#80deea",
    "#ce93d8",
    "#ffcc80",
    "#b0bec5",
    "#bcaaa4",
    "#ffe082",
    "#80cbc4",
    "#c5e1a5",
    "#fff59d",
    "#b39ddb",
    "#81d4fa",
    "#f48fb1",
]

# --------------------------------------------------------------------------- #
# Helpers: TLE fetch & orbit math (identical logic to plot_orbit2d.py)
# --------------------------------------------------------------------------- #


def parse_epoch(s: str) -> datetime:
    for fmt in _EPOCH_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse epoch '{s}'. Expected ISO 8601, e.g. 2027-01-15T06:30:00"
    )


def resolve_constellation(name: str) -> str:
    key = name.strip().lower()
    if key not in _CONSTELLATION_LOOKUP:
        raise ValueError(
            f"Unknown constellation '{name}'. Valid: {', '.join(CONSTELLATIONS)}"
        )
    return _CONSTELLATION_LOOKUP[key]


def fetch_satellites(constellation: str) -> tuple[list[EarthSatellite], object]:
    group, _, name_prefixes = CONSTELLATIONS[constellation]
    url = _CELESTRAK_BASE.format(group=group)
    ts = load.timescale()
    sats = load.tle_file(url, filename=f"tle_{constellation.lower()}.txt")
    if name_prefixes is not None:
        sats = [s for s in sats if any(s.name.startswith(p) for p in name_prefixes)]
    print(f"Loaded {len(sats)} {constellation} satellites from CelesTrak")
    return sats, ts


def fetch_satellite_by_spec(spec: str, ts) -> EarthSatellite:
    spec = spec.strip()
    if spec.isdigit():
        url = _CELESTRAK_CATNR_URL.format(catnr=spec)
        cache = f"tle_catnr_{spec}.txt"
    else:
        url = _CELESTRAK_NAME_URL.format(name=urllib.parse.quote(spec))
        cache = f"tle_name_{spec.replace(' ', '_').replace('/', '_')}.txt"

    try:
        sats = load.tle_file(url, filename=cache)
    except OSError as exc:
        raise ValueError(
            f"Satellite '{spec}' not found (HTTP 404). It may have de-orbited."
        ) from exc

    if not sats:
        raise ValueError(f"No satellite found for '{spec}'.")
    if len(sats) > 1:
        names = ", ".join(s.name for s in sats[:5])
        print(f"  {len(sats)} matches: {names}{'...' if len(sats) > 5 else ''}")
    print(f"  Found: {sats[0].name}")
    return sats[0]


def orbital_period_minutes(sat: EarthSatellite) -> float:
    return 2.0 * math.pi / sat.model.no_kozai


# --------------------------------------------------------------------------- #
# 3-D position computation
# --------------------------------------------------------------------------- #


def compute_gcrs_positions(
    sat: EarthSatellite,
    ts,
    t_epoch: Time,
    duration_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return GCRS orbit trace as (x, y, z) arrays in km."""
    steps = max(360, duration_minutes)
    half = duration_minutes / 2.0
    jds = np.linspace(
        t_epoch.tt - half / 1440.0,
        t_epoch.tt + half / 1440.0,
        steps,
    )
    pos = sat.at(ts.tt_jd(jds)).position.km  # (3, steps)
    return pos[0], pos[1], pos[2]


# --------------------------------------------------------------------------- #
# Plotly scene builders
# --------------------------------------------------------------------------- #


def _earth_surface() -> go.Surface:
    """Blue sphere representing Earth (GCRS frame, no surface texture)."""
    u = np.linspace(0, 2 * np.pi, 180)
    v = np.linspace(0, np.pi, 90)
    x = _RE_KM * np.outer(np.cos(u), np.sin(v))
    y = _RE_KM * np.outer(np.sin(u), np.sin(v))
    z = _RE_KM * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=[[0, "#0d3b6e"], [1, "#1565c0"]],
        showscale=False,
        opacity=0.92,
        lighting={"ambient": 0.5, "diffuse": 0.9, "specular": 0.15, "roughness": 0.7},
        lightposition={"x": 100_000, "y": 80_000, "z": 100_000},
        hoverinfo="skip",
        showlegend=False,
        name="Earth",
    )


def _equator_ring() -> go.Scatter3d:
    theta = np.linspace(0, 2 * np.pi, 360)
    r = _RE_KM * 1.005
    return go.Scatter3d(
        x=r * np.cos(theta),
        y=r * np.sin(theta),
        z=np.zeros(360),
        mode="lines",
        line={"color": "rgba(255,255,255,0.25)", "width": 1},
        hoverinfo="skip",
        showlegend=False,
        name="Equator",
    )


def _pole_axis() -> go.Scatter3d:
    r = _RE_KM * 1.3
    return go.Scatter3d(
        x=[0, 0],
        y=[0, 0],
        z=[-r, r],
        mode="lines",
        line={"color": "rgba(255,255,255,0.2)", "width": 1, "dash": "dash"},
        hoverinfo="skip",
        showlegend=False,
        name="Pole axis",
    )


def _build_layout(title: str, epoch_utc: datetime) -> go.Layout:
    epoch_str = epoch_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    return go.Layout(
        title={
            "text": f"{title}<br><sub>{epoch_str}</sub>",
            "x": 0.5,
            "font": {"size": 15, "color": "white"},
        },
        scene={
            "xaxis": {
                "title": "X [km]",
                "gridcolor": "#2a2a2a",
                "zerolinecolor": "#333",
            },
            "yaxis": {
                "title": "Y [km]",
                "gridcolor": "#2a2a2a",
                "zerolinecolor": "#333",
            },
            "zaxis": {
                "title": "Z [km]",
                "gridcolor": "#2a2a2a",
                "zerolinecolor": "#333",
            },
            "bgcolor": "#080808",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.2, "z": 0.8}},
        },
        paper_bgcolor="#111111",
        font={"color": "#dddddd"},
        legend={
            "bgcolor": "rgba(20,20,20,0.85)",
            "bordercolor": "#444",
            "borderwidth": 1,
            "font": {"size": 9},
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )


# --------------------------------------------------------------------------- #
# Main plot functions
# --------------------------------------------------------------------------- #


def plot_3d_orbits(
    satellites: list[EarthSatellite],
    ts,
    duration_minutes: int,
    epoch_utc: datetime,
    title: str,
    output_path: str | None = None,
) -> None:
    """Render 3-D orbit traces for a list of satellites."""
    t_epoch = ts.from_datetime(epoch_utc)

    traces: list = [_earth_surface(), _equator_ring(), _pole_axis()]

    for idx, sat in enumerate(satellites):
        color = _SAT_COLORS[idx % len(_SAT_COLORS)]

        try:
            x, y, z = compute_gcrs_positions(sat, ts, t_epoch, duration_minutes)
        except (ValueError, RuntimeError) as exc:
            print(f"  Warning: {sat.name} failed: {exc}")
            continue

        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line={"color": color, "width": 1.5},
                name=sat.name,
                hoverinfo="name",
            )
        )

        # Current position marker
        try:
            pos = sat.at(t_epoch).position.km
            traces.append(
                go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode="markers+text",
                    marker={
                        "size": 5,
                        "color": color,
                        "line": {"color": "white", "width": 0.5},
                    },
                    text=[sat.name],
                    textposition="top center",
                    textfont={"size": 7, "color": color},
                    hovertext=(
                        f"{sat.name}<br>"
                        f"x={pos[0]:.0f} km  y={pos[1]:.0f} km  z={pos[2]:.0f} km"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        except (ValueError, RuntimeError):
            pass

    fig = go.Figure(data=traces, layout=_build_layout(title, epoch_utc))
    _save_or_show(fig, output_path)


def _save_or_show(fig: go.Figure, output_path: str | None) -> None:
    if output_path is None:
        output_path = "plot_orbit3d.html"
    if output_path.endswith(".html"):
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"Saved: {output_path}")
    else:
        try:
            fig.write_image(output_path)
            print(f"Saved: {output_path}")
        except (ValueError, OSError) as exc:
            print(f"Cannot save image: {exc}")
            print("Tip: install kaleido with `uv add kaleido` for PNG/SVG output.")
            html_path = output_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(html_path, include_plotlyjs="cdn")
            print(f"Saved as HTML instead: {html_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    valid_names = ", ".join(CONSTELLATIONS)
    parser = argparse.ArgumentParser(
        description="3D satellite orbit visualization (plotly, GCRS inertial frame).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"Supported constellations: {valid_names}\n"
            "Output formats: .html (interactive, default), .png (requires kaleido)\n"
            "\n"
            "Examples:\n"
            "  -c GPS                          GPS constellation\n"
            "  -s STARLINK-1008 -n 3           3 Starlink orbits\n"
            "  -s 44714 -e 2027-06-01T00:00:00 by NORAD ID with epoch"
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
        help="Single satellite by name (e.g. STARLINK-1008) or NORAD ID (e.g. 44714).",
    )
    parser.add_argument(
        "-n",
        "--orbits",
        metavar="N",
        type=int,
        default=1,
        help="Number of orbital periods to show for a single satellite (default: 1). Only used with -s.",
    )
    parser.add_argument(
        "-e",
        "--epoch",
        metavar="DATETIME",
        default=None,
        help="Plot epoch in ISO 8601 UTC (default: 2027-01-15T00:00:00). E.g. 2027-06-01T12:00:00",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        default=None,
        help="Save to file (.html interactive or .png static). Default: plot_orbit3d.html",
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
        period_min = orbital_period_minutes(sat)
        duration = int(period_min * args.orbits)
        orbit_label = f"{args.orbits} orbit{'s' if args.orbits != 1 else ''}"
        title = f"{sat.name}  ({orbit_label}, T={period_min:.1f} min)"
        plot_3d_orbits([sat], ts, duration, epoch_utc, title, args.output)
    else:
        try:
            constellation = resolve_constellation(args.constellation)
        except ValueError as exc:
            parser.error(str(exc))
        _, duration, _ = CONSTELLATIONS[constellation]
        sats, ts = fetch_satellites(constellation)
        plot_3d_orbits(
            sats, ts, duration, epoch_utc, f"{constellation} Constellation", args.output
        )


if __name__ == "__main__":
    main()
