"""Deprecated: moved to misc/gnss_ambiguity.py."""

from misc.gnss_ambiguity import (
    calculate_double_difference,
    calculate_double_difference_ionospheric_ambiguity,
    calculate_double_difference_widelane_ambiguity,
    get_ionospheric_ambiguity,
    get_narrowlane_ambiguity,
    get_widelane_ambiguity,
)

__all__ = [
    "calculate_double_difference",
    "calculate_double_difference_ionospheric_ambiguity",
    "calculate_double_difference_widelane_ambiguity",
    "get_ionospheric_ambiguity",
    "get_narrowlane_ambiguity",
    "get_widelane_ambiguity",
]
