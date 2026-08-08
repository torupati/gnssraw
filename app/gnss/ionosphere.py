from datetime import datetime

import numpy as np

from app.gnss.constants import CLIGHT
from app.gnss.ephemeris import datetime_to_gps_week_seconds


class KlobucharModel:
    """
    Klobuchar ionospheric delay model for GPS.
    Calculates the ionospheric delay using the broadcast
    alpha and beta parameters.
    """

    def __init__(self, alpha: list[float], beta: list[float]):
        """
        Initialize Klobuchar model with broadcast parameters.

        Args:
            alpha: Ionospheric alpha parameters [alpha0, alpha1, alpha2, alpha3]
            beta: Ionospheric beta parameters [beta0, beta1, beta2, beta3]
        """
        if len(alpha) != 4 or len(beta) != 4:
            raise ValueError("Both alpha and beta must have exactly 4 elements.")
        self.alpha = np.array(alpha, dtype=float)
        self.beta = np.array(beta, dtype=float)

    @classmethod
    def from_rinex_nav(cls, nav_header_data: dict) -> "KlobucharModel":
        """
        Create KlobucharModel from a dictionary parsed from RINEX NAV header.
        Assumes nav_header_data has 'ion_alpha' and 'ion_beta' keys, or similar.
        """
        # Adjust key names below based on your actual parsing implementation
        alpha = nav_header_data.get("ion_alpha", [0.0] * 4)
        beta = nav_header_data.get("ion_beta", [0.0] * 4)
        return cls(alpha, beta)

    @classmethod
    def from_almanac(cls, almanac_data: dict) -> "KlobucharModel":
        """
        Create KlobucharModel from external Almanac data if available.
        """
        alpha = almanac_data.get("alpha", [0.0] * 4)
        beta = almanac_data.get("beta", [0.0] * 4)
        return cls(alpha, beta)

    def calculate_delay(
        self,
        dt: datetime,
        receiver_llh: np.ndarray,
        azimuth: float,
        elevation: float,
    ) -> float:
        """
        Calculate ionospheric delay according to the Klobuchar model
        (IS-GPS-200H algorithm).

        Args:
            dt: Observation time
            receiver_llh: Receiver geodetic position [latitude (rad), longitude (rad), altitude (m)]
            azimuth: Satellite azimuth angle in radians (from north)
            elevation: Satellite elevation angle in radians (from horizon)

        Returns:
            Ionospheric delay in meters.
        """
        lat, lon, alt = receiver_llh

        # Return 0.0 if the receiver is underground/at extreme negative altitude or elevation is negative
        if alt < -1e3 or elevation <= 0.0:
            return 0.0

        # Convert to semi-circles
        el_semi = elevation / np.pi
        lat_semi = lat / np.pi
        lon_semi = lon / np.pi

        # 1. Earth-centered angle (semi-circles)
        psi = 0.0137 / (el_semi + 0.11) - 0.022

        # 2. Subionospheric latitude (semi-circles)
        phi_i = lat_semi + psi * np.cos(azimuth)
        phi_i = np.clip(phi_i, -0.416, 0.416)

        # 3. Subionospheric longitude (semi-circles)
        lon_i = lon_semi + (psi * np.sin(azimuth)) / np.cos(phi_i * np.pi)

        # 4. Geomagnetic latitude (semi-circles)
        phi_m = phi_i + 0.064 * np.cos((lon_i - 1.617) * np.pi)

        # 5. Local time (seconds)
        # 1 semi-circle = 12 hours = 43200 seconds
        _, sow = datetime_to_gps_week_seconds(dt)
        t = 43200.0 * lon_i + sow
        t = t % 86400.0  # Keep t in the range [0, 86400)

        # 6. Slant factor
        f = 1.0 + 16.0 * (0.53 - el_semi) ** 3

        # 7 & 9. Calculate amplitude and period (optimization using Horner's method)
        amp = self.alpha[0] + phi_m * (self.alpha[1] + phi_m * (self.alpha[2] + phi_m * self.alpha[3]))
        if amp < 0.0:
            amp = 0.0

        per = self.beta[0] + phi_m * (self.beta[1] + phi_m * (self.beta[2] + phi_m * self.beta[3]))
        if per < 72000.0:
            per = 72000.0

        # 8. Phase of the delay model
        x = 2.0 * np.pi * (t - 50400.0) / per

        # 10. Compute the ionospheric delay in seconds
        if abs(x) < 1.57:
            delay_sec = 5.0e-9 + amp * (1.0 - (x**2) / 2.0 + (x**4) / 24.0)
        else:
            delay_sec = 5.0e-9

        # Apply slant factor and convert to meters
        return CLIGHT * f * delay_sec


class KlobucharManager:
    """
    Manager for storing and retrieving multiple Klobuchar models over time.
    Provides methods to automatically select the most appropriate ionospheric
    params for a given observation epoch.
    """

    def __init__(self):
        self.models: dict[datetime, KlobucharModel] = {}

    def add_model(self, time_of_data: datetime, model: KlobucharModel):
        """
        Add a Klobuchar model valid at or near the given time_of_data.

        Args:
            time_of_data: Timestamp when these parameters were broadcast or the start of their validity
            model: Instance of KlobucharModel
        """
        self.models[time_of_data] = model

    def get_model_for_time(self, obs_time: datetime) -> KlobucharModel | None:
        """
        Finds the closest Klobuchar model preceding the observation time.
        If no preceding model is found, returns the earliest available model.

        Args:
            obs_time: The time of the observation

        Returns:
            The best matching KlobucharModel, or None if no models are available.
        """
        if not self.models:
            return None

        # Try to find the latest model that was broadcast before or at obs_time
        valid_times = [t for t in self.models.keys() if t <= obs_time]

        if valid_times:
            # Get the most recent one among the valid ones
            best_time = max(valid_times)
        else:
            # If all models are strictly after obs_time, fallback to the earliest we have
            best_time = min(self.models.keys())

        return self.models[best_time]
