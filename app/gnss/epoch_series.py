from dataclasses import dataclass
from datetime import datetime

from app.gnss.constants import (
    wlen_L1,
)
from app.gnss.satellite_signals import EpochObservations, SatelliteSignalObservation


@dataclass
class MeasurementForCarrierSmoothing:
    datetime: datetime
    PR: float
    CP: float


def smooth_code_carrier(
    rows_sorted: list[MeasurementForCarrierSmoothing],
    wavelength_m: float,
    code_noise_m: float = 1.0,
    carrier_noise_m: float = 0.02,
    slip_threshold_m: float = 10.0,
    **kwargs,
) -> list[MeasurementForCarrierSmoothing]:
    """Smooth code and carrier phase measurements using a Kalman filter.

    Args:
            rows (list[MeasurementForCarrierSmoothing]): List of measurement objects.
            wavelength_m (float): Wavelength of the carrier signal in meters.
            code_noise_m (float, optional): Standard deviation of code noise in meters. Defaults to 1.0.
            carrier_noise_m (float, optional): Standard deviation of carrier phase noise in meters. Defaults to 0.02.
            slip_threshold_m (float, optional): Threshold for detecting cycle slips in meters. Defaults to 10.0.

    Returns:
            list[MeasurementForCarrierSmoothing]: List of measurement objects with smoothed measurements.
    """
    smoothed_rows: list[MeasurementForCarrierSmoothing] = []

    pr_prev, pr_var = None, 1000.0
    cp_prev = None
    for meas in rows_sorted:
        # print(row)
        pr_obs = meas.PR
        cp = wavelength_m * meas.CP
        # print(f"Raw P: {pr:.3f}, L: {cp:.3f} {pr - wavelength_m * cp:.3f}")
        if pr_prev is None or cp_prev is None:
            pr_prev, cp_prev = pr_obs, cp
            smoothed_rows.append(
                MeasurementForCarrierSmoothing(
                    datetime=meas.datetime,
                    PR=pr_obs,
                    CP=cp,
                )
            )
            continue

        if pr_prev is not None and cp_prev is not None:
            pr_predicted = pr_prev + (cp - cp_prev)
            pr_var = pr_var + 2 * carrier_noise_m**2
            if abs(pr_obs - pr_predicted) > slip_threshold_m:
                msg = (
                    f"{kwargs.get('sat_id', '')} {kwargs.get('band_name', '')} {meas.datetime}: "
                    f"Large prediction error: {pr_obs - pr_predicted:.3f} m, skipping update"
                )
                print(msg)
                pr_prev, pr_var, cp_prev = pr_obs, 1000.0, cp
                smoothed_rows.append(
                    MeasurementForCarrierSmoothing(
                        datetime=meas.datetime,
                        PR=pr_obs,
                        CP=cp,
                    )
                )
                continue
            k_gain = pr_var / (pr_var + code_noise_m**2)
            pr_posteriori = pr_predicted + k_gain * (pr_obs - pr_predicted)
            pr_var_posteriori = (1 - k_gain) * pr_var
            print(
                f"{pr_var=:.3f}, {k_gain:.3f}, {pr_predicted:.3f}, {pr_posteriori:.3f}",
                f" diff={pr_obs - pr_predicted:.3f}",
            )
            # update state
            pr_prev, pr_var, cp_prev = pr_posteriori, pr_var_posteriori, cp
            smoothed_rows.append(
                MeasurementForCarrierSmoothing(
                    datetime=meas.datetime,
                    PR=pr_posteriori,
                    CP=cp,
                )
            )
    return smoothed_rows


def smoothing_code_range_of_receiver(
    epoch_observations: list[EpochObservations],
    code_noise_m: float = 3.0,
    carrier_noise_m: float = 0.01,
    slip_threshold_m: float = 10.0,
) -> list[EpochObservations]:
    """Apply carrier phase smoothing to pseudorange measurements in paired observations.

    Args:
            epoch_observations: List of `EpochObservations` containing epoch and reference observations.
            smoothing_window: The window size for smoothing (in number of epochs).
    """
    if not epoch_observations:
        return epoch_observations

    band_wavelengths = {
        "L1": wlen_L1,
    }

    sat_ids: set[str] = set()
    for epoch_obs in epoch_observations:
        for sat_id, _sat_obs in epoch_obs.iter_satellites():
            sat_ids.add(sat_id)

    sat_band_series: dict[
        str,
        dict[str, list[tuple[MeasurementForCarrierSmoothing, SatelliteSignalObservation]]],
    ] = {}
    for epoch_obs in epoch_observations:
        for sat_id, sat_obs in epoch_obs.iter_satellites():
            if sat_id.startswith("R") or sat_id.startswith("E"):
                continue
            for band_name, signal_obs in sat_obs.signals.items():
                if band_name not in band_wavelengths:
                    continue
                series = sat_band_series.setdefault(sat_id, {}).setdefault(band_name, [])
                series.append(
                    (
                        MeasurementForCarrierSmoothing(
                            datetime=epoch_obs.datetime,
                            PR=signal_obs.pseudorange,
                            CP=signal_obs.carrier_phase,
                        ),
                        signal_obs,
                    )
                )

    for sat_id in sorted(sat_ids):
        band_series = sat_band_series.get(sat_id, {})
        for band_name, series in band_series.items():
            if not series:
                continue
            series_sorted = sorted(series, key=lambda item: item[0].datetime)
            rows_sorted = [item[0] for item in series_sorted]
            smoothed_rows = smooth_code_carrier(
                rows_sorted,
                band_wavelengths[band_name],
                code_noise_m=code_noise_m,
                carrier_noise_m=carrier_noise_m,
                slip_threshold_m=slip_threshold_m,
                sat_id=sat_id,
                band_name=band_name,
            )
            if not smoothed_rows:
                continue
            for (_meas, signal_obs), smoothed in zip(series_sorted, smoothed_rows):
                signal_obs.pseudorange = smoothed.PR
    return epoch_observations
