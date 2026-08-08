from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyrtcm import RTCMReader

from .constants import CLIGHT, GPS_EPOCH, wlen_L1, wlen_L2, wlen_L5, wlen_L7, wlen_L8
from .satellite_signals import (
    EpochObservations,
    SatelliteObservation,
    SatelliteSignalObservation,
)


def _get_attr(msg, *names):
    for name in names:
        if hasattr(msg, name):
            return getattr(msg, name)
    return None


def _get_indexed_values(msg, prefix: str) -> list:
    values = []
    idx = 1
    while True:
        key = f"{prefix}{idx:02d}"
        if not hasattr(msg, key):
            break
        values.append(getattr(msg, key))
        idx += 1
    return values


def _signal_wavelength(sig_code: str) -> float:
    if sig_code.startswith("1"):
        return wlen_L1
    if sig_code.startswith("2"):
        return wlen_L2
    if sig_code.startswith("5"):
        return wlen_L5
    return wlen_L1


def _band_wavelength(band_name: str) -> float:
    """Get wavelength from band name (L1, L2, L5, L7, L8).

    Args:
        band_name: Band name like "L1", "L2", "L5", "L7", "L8"

    Returns:
        Wavelength in meters
    """
    if band_name == "L1":
        return wlen_L1
    elif band_name == "L2":
        return wlen_L2
    elif band_name == "L5":
        return wlen_L5
    elif band_name == "L7":
        return wlen_L7
    elif band_name == "L8":
        return wlen_L8
    else:
        return wlen_L1  # Default to L1


def _signal_code_to_band_name(sig_code: str) -> str:
    """Convert RTCM3 signal code to band name (L1, L2, L5, L7, L8).

    Args:
        sig_code: Signal code like "1C", "2W", "5Q", "7Q", "8Q"

    Returns:
        Band name like "L1", "L2", "L5", "L7", "L8"
    """
    if sig_code[0] in ["1"]:
        return "L1"
    elif sig_code[0] in ["2"]:
        return "L2"
    elif sig_code[0] in ["5"]:
        return "L5"
    elif sig_code[0] in ["7"]:
        return "L7"
    elif sig_code[0] in ["8"]:
        return "L8"
    else:
        return "L1"  # Default to L1


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _iter_msm_cells(msg):
    sats = _get_attr(msg, "sats", "satellites", "sat", "satlist")
    sigs = _get_attr(msg, "sigs", "signals", "sig", "siglist")
    cellmask = _get_attr(msg, "cellmask", "cell_mask", "cell")

    if not sats or not sigs:
        return []

    sats = list(sats)
    sigs = list(sigs)
    total = len(sats) * len(sigs)

    if cellmask is not None:
        cellmask = list(cellmask)
        if len(cellmask) == total:
            cells = []
            for idx, flag in enumerate(cellmask):
                if not flag:
                    continue
                sat = sats[idx // len(sigs)]
                sig = sigs[idx % len(sigs)]
                cells.append((sat, sig))
            return cells

    return [(sat, sig) for sat in sats for sig in sigs]


def parse_rtcm_msm7_signal_observations(msg):
    cell_prn = _get_indexed_values(msg, "CELLPRN_")
    cell_sig = _get_indexed_values(msg, "CELLSIG_")
    pr_list = _get_indexed_values(msg, "DF404_")
    cp_list = _get_indexed_values(msg, "DF405_")
    dp_list = _get_indexed_values(msg, "DF406_")
    cnr_list = _get_indexed_values(msg, "DF408_")
    rough_range_list = _get_indexed_values(msg, "DF397_")
    rough_range_mod_list = _get_indexed_values(msg, "DF398_")
    # NOTE: Immediately cache rough_rate to avoid pyrtcm field mutation
    rough_rate_list = _get_indexed_values(msg, "DF399_")

    # Get satellite list directly from PRN_xx fields
    prn_list = _get_indexed_values(msg, "PRN_")

    # Build map: PRN -> (rough_range, rough_range_mod, rough_rate)
    rough_by_prn = {}
    for sat_idx, prn in enumerate(prn_list):
        rough = float(rough_range_list[sat_idx]) if sat_idx < len(rough_range_list) else 0.0
        rough_mod = float(rough_range_mod_list[sat_idx]) if sat_idx < len(rough_range_mod_list) else 0.0
        rough_rate = float(rough_rate_list[sat_idx]) if sat_idx < len(rough_rate_list) else 0.0
        rough_by_prn[int(prn)] = (rough, rough_mod, rough_rate)

    count = min(
        len(cell_prn),
        len(cell_sig),
        len(pr_list),
        len(cp_list),
        len(dp_list),
        len(cnr_list),
    )

    # Remove debug code
    if count == 0:
        cells = _iter_msm_cells(msg)
        pr_list = _ensure_list(_get_attr(msg, "fine_pseudorange", "pseudorange", "pr"))
        cp_list = _ensure_list(_get_attr(msg, "fine_phaserange", "phaserange", "carrier_phase", "cp"))
        dp_list = _ensure_list(_get_attr(msg, "phaserange_rate", "phaserate", "doppler", "rate"))
        cnr_list = _ensure_list(_get_attr(msg, "cnr", "snr", "CNR"))
        count = min(
            len(cells),
            len(pr_list),
            len(cp_list),
            len(dp_list),
            len(cnr_list),
        )
        observations = []
        for idx in range(count):
            sat, sig = cells[idx]
            wavelength = _signal_wavelength(str(sig))
            obs = SatelliteSignalObservation(
                pseudorange=float(pr_list[idx]),
                carrier_phase=float(cp_list[idx]) / wavelength,
                doppler_=-float(dp_list[idx]) / wavelength,
                snr=float(cnr_list[idx]),
            )
            observations.append((sat, sig, obs))
        return observations

    observations = []
    for idx in range(count):
        sat = int(cell_prn[idx])
        sig = str(cell_sig[idx])
        wavelength = _signal_wavelength(sig)
        rough_range_ms, rough_mod_ms, rough_rate = rough_by_prn.get(sat, (0.0, 0.0, 0.0))
        rough_range_m = (rough_range_ms + rough_mod_ms) * 1e-3 * CLIGHT
        pseudorange_m = rough_range_m + float(pr_list[idx]) * 0.0001

        # DF405 is in wavelength units, so convert to phase_range in meters first
        # then to cycles
        fine_phase_wavelengths = float(cp_list[idx])
        phase_range_m = rough_range_m + fine_phase_wavelengths * wavelength

        # Combine rough rate (DF399 in m/s) with fine rate (DF406 in 0.0001 m/s)
        range_rate_mps = float(rough_rate) + float(dp_list[idx]) * 0.0001
        obs = SatelliteSignalObservation(
            pseudorange=pseudorange_m,
            carrier_phase=phase_range_m / wavelength,
            doppler_=-range_rate_mps / wavelength,
            snr=float(cnr_list[idx]),
        )
        observations.append((sat, sig, obs))
    return observations


def group_observations_by_satellite(
    observations: list[tuple[int, str, SatelliteSignalObservation]],
) -> list[SatelliteObservation]:
    by_prn: dict[int, SatelliteObservation] = {}
    for prn, sig, obs in observations:
        if prn not in by_prn:
            by_prn[prn] = SatelliteObservation(prn=prn, signals={}, ambiguities={})
        # Convert signal code (e.g., "1C") to band name (e.g., "L1")
        band_name = _signal_code_to_band_name(str(sig))
        by_prn[prn].signals[band_name] = obs
    return [by_prn[prn] for prn in sorted(by_prn)]


def gps_tow_to_datetime(gps_week: int, tow_ms: float) -> datetime:
    tow_seconds = tow_ms / 1000.0
    dt = GPS_EPOCH + timedelta(weeks=gps_week, seconds=tow_seconds)
    return dt.replace(tzinfo=None)


def read_rtcm3_file(rtcm_path: Path, gps_week_number: int) -> list[EpochObservations]:
    """
    Read RTCM3 MSM7 observation messages and return a list of unified EpochObservations.

    Supports:
    - RTCM 1077 (GPS)
    - RTCM 1097 (Galileo)
    - RTCM 1117 (QZSS)

    Observations from the same time-of-week are grouped into a single EpochObservations.

    Args:
        rtcm_path: Path to RTCM3 binary file
        gps_week_number: GPS week number for converting TOW to datetime
    Returns:
        List of EpochObservations, each containing synchronized multi-constellation observations
    """
    if not rtcm_path.exists():
        raise FileNotFoundError(f"RTCM3 file not found: {rtcm_path}")

    epochs: list[EpochObservations] = []
    buffered_epoch: EpochObservations | None = None
    last_tow: int | None = None

    with rtcm_path.open("rb") as stream:
        for _raw, msg in RTCMReader(stream):
            if msg is None:
                continue
            identity = str(msg.identity)
            if identity not in {"1077", "1117", "1097"}:
                continue

            # Get TOW: DF004 for GPS, DF248 for Galileo, DF428 for QZSS
            tow_ms = _get_attr(msg, "DF004") or _get_attr(msg, "DF248") or _get_attr(msg, "DF428")

            # Parse observations
            observations: list[tuple[int, str, SatelliteSignalObservation]] = parse_rtcm_msm7_signal_observations(msg)
            if not observations:
                continue

            # Convert TOW to datetime
            epoch_dt = None
            if tow_ms is not None:
                try:
                    tow_value = float(tow_ms)
                    epoch_dt = gps_tow_to_datetime(gps_week_number, tow_value)
                except (TypeError, ValueError):
                    epoch_dt = None
            if epoch_dt is None:
                epoch_dt = datetime.fromtimestamp(0, tz=timezone.utc).replace(tzinfo=None)

            # Group observations by satellite
            satellites = group_observations_by_satellite(observations)

            # Check if this timestamp matches the buffered epoch
            current_tow = int(tow_ms) if tow_ms is not None else None

            if current_tow is not None and last_tow is not None and current_tow == last_tow and buffered_epoch is not None:
                # Same TOW: merge into buffered epoch
                if identity == "1077":
                    buffered_epoch.satellites_gps.extend(satellites)
                elif identity == "1117":
                    buffered_epoch.satellites_qzss.extend(satellites)
                elif identity == "1097":
                    buffered_epoch.satellites_galileo.extend(satellites)
            else:
                # Different TOW or first message: flush buffered epoch if exists
                if buffered_epoch is not None:
                    epochs.append(buffered_epoch)

                # Create new buffered epoch
                satellites_gps = satellites if identity == "1077" else []
                satellites_qzss = satellites if identity == "1117" else []
                satellites_galileo = satellites if identity == "1097" else []
                buffered_epoch = EpochObservations(
                    datetime=epoch_dt,
                    satellites_gps=satellites_gps,
                    satellites_qzss=satellites_qzss,
                    satellites_galileo=satellites_galileo,
                    satellites_glonass=[],
                )
                last_tow = current_tow

        # Flush the final buffered epoch
        if buffered_epoch is not None:
            epochs.append(buffered_epoch)

    return epochs
