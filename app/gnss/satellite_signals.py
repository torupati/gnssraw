import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path

import georinex as gr
import numpy as np

from app.gnss.constants import (
    CLIGHT,
    E5B_FREQ,
    E8_FREQ,
    L1_FREQ,
    L2_FREQ,
    L5_FREQ,
    wlen_L1,
    wlen_L2,
    wlen_L5,
    wlen_L7,
    wlen_L8,
)

logger = getLogger(__name__)

# Suppress FutureWarnings from xarray/georinex
warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")


@dataclass
class SatelliteSignalObservation:
    """Observation data for a single satellite signal band.
    Attributes:
        pseudorange: Pseudorange measurement in meters
        carrier_phase: Carrier phase measurement in cycles
        doppler_: Doppler measurement in Hz
        snr: Signal-to-noise ratio in dB-Hz
    """

    pseudorange: float  # in meters
    carrier_phase: float  # in cycles
    doppler_: float  # in Hz
    snr: float  # in dB-Hz

    def __call__(self, *args, **kwds):
        pass

    def __str__(self):
        return (
            f"SatelliteSignalObservation(pseudorange={self.pseudorange:.3f} m, "
            f"carrier_phase={self.carrier_phase:.3f} cycles, "
            f"doppler={self.doppler_:.3f} Hz, "
            f"snr={self.snr:.1f} dB-Hz)"
        )


@dataclass
class AmbiguityObservation:
    """Ambiguity observations for dual-frequency combinations.
    Attributes:
        widelane: Widelane ambiguity in cycles
        ionofree: Ionosphere-free ambiguity in cycles
        geofree: Geometry-free ambiguity in cycles (optional, defaults to 0.0)
    """

    widelane: float  # in cycle
    ionofree: float  # in cycle
    geofree: float = 0.0  # in cycle (optional, default to 0.0)
    multipath: float = 0.0  # in meters (optional, default to 0.0)


@dataclass
class SatelliteObservation:
    """GNSS observations for a single satellite.
    Attributes:
        prn: Satellite PRN number
        signals: Dict of band name to SatelliteSignalObservation
        ambiguities: Dict of combination name to AmbiguityObservation
    """

    prn: int
    signals: dict  # key: band (str), value: SatelliteSignalObservation
    ambiguities: dict  # key: combination (str like "L1_L2"), value: AmbiguityObservation

    def add_signal_observation(self, band_name: str, signal_obs: SatelliteSignalObservation):
        self.signals[band_name] = signal_obs

    def __str__(self):
        return f"SatelliteObservation(prn={self.prn}, signals={list(self.signals.keys())}, ambiguities={list(self.ambiguities.keys())})"


@dataclass
class EpochObservations:
    """GNSS observations for a single epoch.
    Attributes:
        datetime: Observation epoch time (UTC)
        satellites_gps: List of SatelliteObservation for GPS satellites
        satellites_qzss: List of SatelliteObservation for QZSS satellites
        satellites_galileo: List of SatelliteObservation for Galileo satellites
        satellites_glonass: List of SatelliteObservation for GLONASS satellites
    """

    datetime: datetime
    satellites_gps: list[SatelliteObservation]
    satellites_qzss: list[SatelliteObservation]
    satellites_galileo: list[SatelliteObservation]
    satellites_glonass: list[SatelliteObservation]

    def __str__(self):
        return (
            f"EpochObservations(datetime={self.datetime}, "
            f"GPS_sats={len(self.satellites_gps)}, "
            f"QZSS_sats={len(self.satellites_qzss)}, "
            f"Galileo_sats={len(self.satellites_galileo)}, "
            f"GLONASS_sats={len(self.satellites_glonass)})"
        )

    def iter_satellites(self):
        """Iterate over all satellites in the epoch, yielding (sat_id, sat_obs)."""
        for sat_list, system_code in [
            (self.satellites_gps, "G"),
            (self.satellites_qzss, "J"),
            (self.satellites_galileo, "E"),
            (self.satellites_glonass, "R"),
        ]:
            for sat_obs in sat_list:
                sat_id = f"{system_code}{sat_obs.prn:02d}"
                yield sat_id, sat_obs


@dataclass
class PairedObservation:
    """Paired GNSS observations for a single epoch.
    This is used for comparing two sets of observations (e.g., from two receivers).
    Attributes:
        epoch: Epoch identifier as a string
        datetime: Observation epoch time (UTC)
        observation: Primary EpochObservations
        ref_observation: Reference EpochObservations
        combined_observations: Optional list of combined observation dicts
    """

    epoch: str
    datetime: datetime
    observation: EpochObservations
    ref_observation: EpochObservations
    combined_observations: list[dict] | None = None  # list of combined observation dicts

    @property
    def time_str(self) -> str:
        return self.datetime.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def age_seconds(self) -> float:
        return (self.datetime - self.ref_observation.datetime).total_seconds()


def compute_dual_frequency_ambiguity(
    pr_f1: float,
    cp_f1: float,
    pr_f2: float,
    cp_f2: float,
    freq1: float,
    freq2: float,
    wlen1: float,
    wlen2: float,
) -> AmbiguityObservation:
    """
    Compute widelane and ionofree ambiguity for dual-frequency observations.

    Args:
        pr_f1: Pseudorange for frequency 1 (meters)
        cp_f1: Carrier phase for frequency 1 (cycles)
        pr_f2: Pseudorange for frequency 2 (meters)
        cp_f2: Carrier phase for frequency 2 (cycles)
        freq1: Frequency 1 (Hz)
        freq2: Frequency 2 (Hz)
        wlen1: Wavelength 1 (meters)
        wlen2: Wavelength 2 (meters)
    Returns:
        AmbiguityObservation with widelane and ionofree ambiguities in cycles
    """
    # Compute MW combination (cycles)
    wl_wlen = CLIGHT / (freq1 - freq2)
    cp_wl = cp_f1 - cp_f2
    pr_nl = (freq1 / (freq1 + freq2)) * pr_f1 + (freq2 / (freq1 + freq2)) * pr_f2
    amb_wl = cp_wl - pr_nl / wl_wlen  # geometry-free and ionosphere-free ambiguity (N1 - N2)

    # Compute ionofree ambiguity (cycles)
    pr_if = (freq1**2 * pr_f1 - freq2**2 * pr_f2) / (freq1**2 - freq2**2)
    cp_if = (freq1**2 * wlen1 * cp_f1 - freq2**2 * wlen2 * cp_f2) / (freq1**2 - freq2**2)
    amb_iono = (cp_if - pr_if) / ((freq1**2) / (freq1**2 - freq2**2) * wlen1 + (freq2**2) / (freq1**2 - freq2**2) * wlen2)

    # Compute geometry-free ambiguity (cycles)
    amb_geofree = (cp_f1 * wlen1 - cp_f2 * wlen2) / (wlen1 - wlen2) - np.round(amb_wl) / (wlen1 - wlen2)

    # Compute multipath estimate (meters) using geometry-free combination
    mp_estimate = (pr_f1 - wlen1 * cp_f1) + wlen1**2 * (wlen2 * cp_f2 - wlen1 * cp_f1) / (wlen2**2 - wlen1**2)

    return AmbiguityObservation(widelane=amb_wl, ionofree=amb_iono, geofree=amb_geofree, multipath=mp_estimate)


def compute_ambiguities_for_satellite(sat_obs: SatelliteObservation, system_name: str) -> dict:
    """
    Compute widelane and ionofree ambiguities for all available frequency pairs.

    For GPS/QZSS: Compute L1/L2 and L1/L5 if available
    For Galileo: Compute L1/L5, L1/L7, L1/L8 if available

    Args:
        sat_obs: SatelliteObservation with signals dict
        system_name: System name ("GPS", "QZSS", or "Galileo")
    Returns:
        Dict of {combination_name: AmbiguityObservation}
    """
    ambiguities = {}

    # GPS/QZSS: L1/L2 combination
    if system_name in ["GPS", "QZSS"]:
        if "L1" in sat_obs.signals and "L2" in sat_obs.signals:
            sig_l1 = sat_obs.signals["L1"]
            sig_l2 = sat_obs.signals["L2"]
            ambiguities["L1_L2"] = compute_dual_frequency_ambiguity(
                sig_l1.pseudorange,
                sig_l1.carrier_phase,
                sig_l2.pseudorange,
                sig_l2.carrier_phase,
                L1_FREQ,
                L2_FREQ,
                wlen_L1,
                wlen_L2,
            )

    # GPS/QZSS/Galileo: L1/L5 combination
    if system_name in ["GPS", "QZSS", "Galileo"]:
        if "L1" in sat_obs.signals and "L5" in sat_obs.signals:
            sig_l1 = sat_obs.signals["L1"]
            sig_l5 = sat_obs.signals["L5"]
            ambiguities["L1_L5"] = compute_dual_frequency_ambiguity(
                sig_l1.pseudorange,
                sig_l1.carrier_phase,
                sig_l5.pseudorange,
                sig_l5.carrier_phase,
                L1_FREQ,
                L5_FREQ,
                wlen_L1,
                wlen_L5,
            )

    # Galileo: L1/L7(E1/E5B), L1/L8(E1/E5AB) combination
    if system_name == "Galileo":
        if "L1" in sat_obs.signals and "L7" in sat_obs.signals:
            sig_l1 = sat_obs.signals["L1"]
            sig_l7 = sat_obs.signals["L7"]
            ambiguities["L1_L7"] = compute_dual_frequency_ambiguity(
                sig_l1.pseudorange,
                sig_l1.carrier_phase,
                sig_l7.pseudorange,
                sig_l7.carrier_phase,
                L1_FREQ,
                E5B_FREQ,
                wlen_L1,
                wlen_L7,
            )

        # Galileo: L1/L8 combination (E1/E8)
        if "L1" in sat_obs.signals and "L8" in sat_obs.signals:
            sig_l1 = sat_obs.signals["L1"]
            sig_l8 = sat_obs.signals["L8"]
            ambiguities["L1_L8"] = compute_dual_frequency_ambiguity(
                sig_l1.pseudorange,
                sig_l1.carrier_phase,
                sig_l8.pseudorange,
                sig_l8.carrier_phase,
                L1_FREQ,
                E8_FREQ,
                wlen_L1,
                wlen_L8,
            )

    return ambiguities


def calculate_combined_observations(
    obss: list[EpochObservations],
) -> list[EpochObservations]:
    """Calculate combined observations from multiple epochs. This can be used to combine observations from two receivers or to create averaged observations.

    Args:
        obss (list[EpochObservations]): List of EpochObservations to combine

    Returns:
        list[EpochObservations]: List of combined EpochObservations
    """
    for obs in obss:
        for sat_obs in obs.satellites_gps:
            sat_obs.ambiguities = compute_ambiguities_for_satellite(sat_obs, "GPS")
        for sat_obs in obs.satellites_qzss:
            sat_obs.ambiguities = compute_ambiguities_for_satellite(sat_obs, "QZSS")
        for sat_obs in obs.satellites_galileo:
            sat_obs.ambiguities = compute_ambiguities_for_satellite(sat_obs, "Galileo")
    return obss


def parse_rinex_observation_file(
    file_path: str,
    signal_code_map: dict[str, list[list[str]]],
) -> list[EpochObservations]:
    """
    Parse a RINEX observation file and return a list of EpochObservations.

    Args:
        file_path: Path to the RINEX observation file.
        signal_code_map: Mapping of system name to list of [band_number, signal_code]
            pairs for observation codes like C1C, L1C, C5X, L5X.
    Returns:
        List of EpochObservations containing satellite observations per epoch.
    """
    # Placeholder for actual parsing logic
    epochs = []
    try:
        _data = gr.load(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load RINEX file: {e}")
    # Implement parsing logic here to populate epochs
    for time_idx, time_val in enumerate(_data.time.values):
        # Convert numpy.datetime64 to seconds since Unix epoch, then to a naive UTC datetime
        timestamp_seconds = time_val.astype("datetime64[s]").astype("int64")
        epoch_obs = EpochObservations(
            datetime=datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc).replace(tzinfo=None),
            satellites_gps=[],
            satellites_qzss=[],
            satellites_galileo=[],
            satellites_glonass=[],
        )
        # Iterate over satellites and populate epoch_obs
        sv_values = [str(s) for s in _data.sv.values]
        for sv in sv_values:
            prn = int(sv[1:]) if sv[1:].isdigit() else None
            if prn is None:
                continue
            sat_obs = SatelliteObservation(prn=prn, signals={}, ambiguities={})
            # GPS (G)
            for system_code, system_name in [
                ("G", "GPS"),
                ("J", "QZSS"),
                ("E", "Galileo"),
                ("R", "GLONASS"),
            ]:
                if not sv.startswith(system_code):
                    continue
                if system_name not in signal_code_map:
                    raise KeyError(f"signal_code_map missing system: {system_name}")
                # Extract signal observations (pseudorange, carrier phase, etc.)
                for band_number, signal_code in signal_code_map[system_name]:
                    pr_key = f"C{band_number}{signal_code}"
                    cp_key = f"L{band_number}{signal_code}"
                    dp_key = f"D{band_number}{signal_code}"
                    snr_key = f"S{band_number}{signal_code}"

                    # Check if all required keys exist in the dataset
                    if not all(key in _data for key in [pr_key, cp_key, dp_key, snr_key]):
                        continue

                    pr = _data[pr_key].sel(sv=sv, time=time_val).values
                    cp = _data[cp_key].sel(sv=sv, time=time_val).values
                    dp = _data[dp_key].sel(sv=sv, time=time_val).values
                    snr = _data[snr_key].sel(sv=sv, time=time_val).values

                    # Skip if any value is NaN or size is 0
                    if pr.size > 0 and cp.size > 0 and dp.size > 0 and snr.size > 0:
                        if not (np.isnan(pr).any() or np.isnan(cp).any() or np.isnan(dp).any() or np.isnan(snr).any()):
                            signal_obs = SatelliteSignalObservation(
                                pseudorange=float(pr),
                                carrier_phase=float(cp),
                                doppler_=float(dp),
                                snr=float(snr),
                            )
                            # Use band name like L1, L2, L5, L7, L8
                            band_name = f"L{band_number}"
                            sat_obs.add_signal_observation(band_name, signal_obs)
                # Only add satellite observation if it has signals
                if sat_obs.signals:
                    # Compute ambiguities for GPS/QZSS/Galileo
                    if system_name in ["GPS", "QZSS", "Galileo"]:
                        sat_obs.ambiguities = compute_ambiguities_for_satellite(sat_obs, system_name)

                    if system_name == "GPS":
                        epoch_obs.satellites_gps.append(sat_obs)
                    elif system_name == "QZSS":
                        epoch_obs.satellites_qzss.append(sat_obs)
                    elif system_name == "Galileo":
                        epoch_obs.satellites_galileo.append(sat_obs)
                    elif system_name == "GLONASS":
                        epoch_obs.satellites_glonass.append(sat_obs)
            # Similar logic for other constellations (QZSS, Galileo, GLONASS)
        epochs.append(epoch_obs)
    return epochs


def compute_ambiguity_statistics(
    epochs: list[EpochObservations],
) -> list[dict]:
    """
    Compute statistics for AmbiguityObservation data across all epochs.

    Args:
        epochs: List of EpochObservations

    Returns:
        List of dictionaries containing statistics for each satellite-band combination:
        - start_time: First observation datetime for this satellite-band
        - end_time: Last observation datetime for this satellite-band
        - satellite: Satellite ID (e.g., "G01")
        - band: Frequency combination (e.g., "L1_L2")
        - num_epochs: Number of observations
        - widelane_ambiguity_mean: Mean of widelane ambiguity
        - widelane_ambiguity_std: Standard deviation of widelane ambiguity
        - widelane_ambiguity_max_min: Range (max - min) of widelane ambiguity
        - snr_mean: Mean SNR for the first band in combination
        - snr_max_min: Range (max - min) of SNR
    """
    # Organize data by satellite and band combination
    satellite_ambiguity_data: dict[str, dict[str, dict]] = {}

    for epoch in epochs:
        for sat_list, system_code in [
            (epoch.satellites_gps, "G"),
            (epoch.satellites_qzss, "J"),
            (epoch.satellites_galileo, "E"),
            (epoch.satellites_glonass, "R"),
        ]:
            for sat_obs in sat_list:
                sat_id = f"{system_code}{sat_obs.prn:02d}"

                if sat_id not in satellite_ambiguity_data:
                    satellite_ambiguity_data[sat_id] = {}

                # Process each ambiguity combination
                for comb_name, amb_obs in sat_obs.ambiguities.items():
                    if comb_name not in satellite_ambiguity_data[sat_id]:
                        satellite_ambiguity_data[sat_id][comb_name] = {
                            "times": [],
                            "widelane_values": [],
                            "snr_values": [],
                        }

                    # Record epoch time
                    satellite_ambiguity_data[sat_id][comb_name]["times"].append(epoch.datetime)

                    satellite_ambiguity_data[sat_id][comb_name]["widelane_values"].append(amb_obs.widelane)

                    # Extract SNR from the first band in the combination
                    # For "L1_L2", get SNR from L1 band
                    first_band = comb_name.split("_")[0]
                    if first_band in sat_obs.signals:
                        satellite_ambiguity_data[sat_id][comb_name]["snr_values"].append(sat_obs.signals[first_band].snr)

    # Compute statistics for each satellite-band combination
    statistics = []
    for sat_id, band_data in satellite_ambiguity_data.items():
        for band, values in band_data.items():
            times = values["times"]
            wl_values = values["widelane_values"]
            snr_values = values["snr_values"]

            # Get start and end times for this satellite-band combination
            start_time = min(times) if times else None
            end_time = max(times) if times else None

            if wl_values:
                wl_mean = sum(wl_values) / len(wl_values)
                wl_std = (sum((x - wl_mean) ** 2 for x in wl_values) / len(wl_values)) ** 0.5
                wl_max_min = max(wl_values) - min(wl_values)
            else:
                wl_mean = wl_std = wl_max_min = 0.0

            if snr_values:
                snr_mean = sum(snr_values) / len(snr_values)
                snr_max_min = max(snr_values) - min(snr_values)
            else:
                snr_mean = snr_max_min = 0.0

            statistics.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "satellite": sat_id,
                    "band": band,
                    "num_epochs": len(wl_values),
                    "widelane_ambiguity_mean": wl_mean,
                    "widelane_ambiguity_std": wl_std,
                    "widelane_ambiguity_max_min": wl_max_min,
                    "snr_mean": snr_mean,
                    "snr_max_min": snr_max_min,
                }
            )

    return statistics


def save_ambiguity_statistics_to_csv(
    epochs: list[EpochObservations],
    output_file: Path,
):
    """
    Save ambiguity statistics to CSV file.

    Args:
        epochs: List of EpochObservations
        output_file: Path to output CSV file
    """
    import csv

    statistics = compute_ambiguity_statistics(epochs)

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write statistics to CSV
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        if statistics:
            fieldnames = [
                "start_time",
                "end_time",
                "satellite",
                "band",
                "num_epochs",
                "widelane_ambiguity_mean",
                "widelane_ambiguity_std",
                "widelane_ambiguity_max_min",
                "snr_mean",
                "snr_max_min",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Convert datetime objects to ISO format strings
            for stat in statistics:
                if stat["start_time"] is not None:
                    stat["start_time"] = stat["start_time"].isoformat()
                if stat["end_time"] is not None:
                    stat["end_time"] = stat["end_time"].isoformat()
            writer.writerows(statistics)

    logger.info(f"Saved ambiguity statistics to CSV: {output_file}")


def save_gnss_observations_to_json(
    epochs: list[EpochObservations],
    output_file: str,
):
    # Convert to JSON-serializable format
    def convert_to_json_serializable(obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            return convert_to_json_serializable(obj.__dict__)
        else:
            return obj

    import json

    output_data = {
        "filename": str(output_file),
        "epochs": [convert_to_json_serializable(epoch) for epoch in epochs],
    }
    json_str = json.dumps(output_data, indent=2)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json_str)
