from bisect import bisect_left
from datetime import datetime
from logging import getLogger

from app.gnss.satellite_signals import (
    EpochObservations,
    PairedObservation,
    SatelliteObservation,
)

logger = getLogger(__name__)


def _find_closest_epoch(
    ref_epochs: list[EpochObservations],
    ref_times: list[datetime],
    target_time: datetime,
) -> EpochObservations:
    if not ref_epochs:
        raise ValueError("Reference epochs are empty")
    idx = bisect_left(ref_times, target_time)
    if idx == 0:
        return ref_epochs[0]
    if idx >= len(ref_times):
        return ref_epochs[-1]
    before = ref_epochs[idx - 1]
    after = ref_epochs[idx]
    if abs((target_time - before.datetime).total_seconds()) <= abs((after.datetime - target_time).total_seconds()):
        return before
    return after


def pair_observations(
    epochs: list[EpochObservations],
    ref_epochs: list[EpochObservations],
) -> list[PairedObservation]:
    """Pair epochs with nearest reference epochs.

    Args:
            epochs: Epoch observations to be paired.
            ref_epochs: Reference epoch observations used for pairing.

    Returns:
            List of `PairedObservation` with matched reference epochs.

    Raises:
            ValueError: If `ref_epochs` is empty.
    """
    ref_times = [epoch.datetime for epoch in ref_epochs]
    paired: list[PairedObservation] = []
    for epoch in epochs:
        ref_epoch = _find_closest_epoch(ref_epochs, ref_times, epoch.datetime)
        paired.append(
            PairedObservation(
                epoch=epoch.datetime.isoformat(),
                datetime=epoch.datetime,
                observation=epoch,
                ref_observation=ref_epoch,
                combined_observations=[],
            )
        )
    return paired


def _build_satellite_obs_map(
    epoch: EpochObservations,
) -> dict[str, SatelliteObservation]:
    sat_map: dict[str, SatelliteObservation] = {}
    for sat_id, sat_obs in epoch.iter_satellites():
        sat_map[sat_id] = sat_obs
    return sat_map


def _compute_double_difference_for_pair(
    observation: EpochObservations,
    ref_observation: EpochObservations,
    sat1: str,
    sat2: str,
    combination: str,
) -> dict:
    obs_map = _build_satellite_obs_map(observation)
    ref_map = _build_satellite_obs_map(ref_observation)

    obs_sat1 = obs_map.get(sat1)
    obs_sat2 = obs_map.get(sat2)
    ref_sat1 = ref_map.get(sat1)
    ref_sat2 = ref_map.get(sat2)

    if not (obs_sat1 and obs_sat2 and ref_sat1 and ref_sat2):
        return {
            "sat1": sat1,
            "sat2": sat2,
            "combination": combination,
            "widelane": None,
            "ionofree": None,
        }

    obs_amb1 = obs_sat1.ambiguities.get(combination)
    obs_amb2 = obs_sat2.ambiguities.get(combination)
    ref_amb1 = ref_sat1.ambiguities.get(combination)
    ref_amb2 = ref_sat2.ambiguities.get(combination)

    if not (obs_amb1 and obs_amb2 and ref_amb1 and ref_amb2):
        return {
            "sat1": sat1,
            "sat2": sat2,
            "combination": combination,
            "widelane": None,
            "ionofree": None,
        }

    widelane_dd = (obs_amb1.widelane - obs_amb2.widelane) - (ref_amb1.widelane - ref_amb2.widelane)
    ionofree_dd = (obs_amb1.ionofree - obs_amb2.ionofree) - (ref_amb1.ionofree - ref_amb2.ionofree)

    return {
        "sat1": sat1,
        "sat2": sat2,
        "combination": combination,
        "widelane": widelane_dd,
        "ionofree": ionofree_dd,
    }


def update_combined_observation(
    paired_epochs: list[PairedObservation],
    sat_pairs: list[dict],
):
    """Update paired epochs with double-difference combined observations.

    For each `PairedObservation`, this computes double-difference widelane and
    ionofree ambiguities for satellite pairs defined by `sat_pairs`, and stores
    results in `pair.combined_observations`.

    Args:
            paired_epochs: Paired epochs to update.
            sat_pairs: List of dictionaries with keys `sat1`, `sat2`, and optional
                    `combinations` (list of ambiguity combination names to include).
    """
    for pair in paired_epochs:
        combined = []
        obs_map = _build_satellite_obs_map(pair.observation)
        ref_map = _build_satellite_obs_map(pair.ref_observation)
        for entry in sat_pairs:
            sat1 = entry["sat1"]
            sat2 = entry["sat2"]
            combinations = entry.get("combinations")

            obs_sat1 = obs_map.get(sat1)
            obs_sat2 = obs_map.get(sat2)
            ref_sat1 = ref_map.get(sat1)
            ref_sat2 = ref_map.get(sat2)

            if not (obs_sat1 and obs_sat2 and ref_sat1 and ref_sat2):
                logger.warning(
                    "Satellites %s or %s not found in observations at %s",
                    sat1,
                    sat2,
                    pair.datetime,
                )
                continue

            obs_keys = set(obs_sat1.ambiguities.keys()) & set(obs_sat2.ambiguities.keys())
            ref_keys = set(ref_sat1.ambiguities.keys()) & set(ref_sat2.ambiguities.keys())
            common_keys = sorted(obs_keys & ref_keys)

            if combinations is not None:
                common_keys = [key for key in common_keys if key in combinations]

            if not common_keys:
                logger.warning(
                    "No common ambiguity combinations for satellites %s and %s at %s",
                    sat1,
                    sat2,
                    pair.datetime,
                )
                continue

            for comb in common_keys:
                combined.append(_compute_double_difference_for_pair(pair.observation, pair.ref_observation, sat1, sat2, comb))

        pair.combined_observations = combined
