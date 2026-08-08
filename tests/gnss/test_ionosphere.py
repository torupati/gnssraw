from datetime import datetime

import numpy as np

from app.gnss.ionosphere import KlobucharModel


def test_klobuchar_model_delay():
    """
    Test Klobuchar delay calculation using parameters from a typical RINEX 3.02 nav header:
    GPSA   1.4901D-08  1.4901D-08 -5.9605D-08 -5.9605D-08       IONOSPHERIC CORR
    GPSB   9.2160D+04  4.9152D+04 -1.3107D+05 -3.2768D+05       IONOSPHERIC CORR
    """
    alpha = [1.4901e-08, 1.4901e-08, -5.9605e-08, -5.9605e-08]
    beta = [9.2160e04, 4.9152e04, -1.3107e05, -3.2768e05]

    model = KlobucharModel(alpha=alpha, beta=beta)

    # Mock observation time (e.g., matching the data timeframe)
    dt = datetime(2025, 12, 24, 22, 0, 0)

    # Receiver approximate position (e.g., Tokyo: 35.68 N, 139.76 E, 50m altitude)
    lat_rad = np.radians(35.68)
    lon_rad = np.radians(139.76)
    alt = 50.0
    receiver_llh = np.array([lat_rad, lon_rad, alt])

    # Satellite observed at 45 deg azimuth and 30 deg elevation
    azimuth = np.radians(45.0)
    elevation = np.radians(30.0)

    delay_m = model.calculate_delay(
        dt=dt, receiver_llh=receiver_llh, azimuth=azimuth, elevation=elevation
    )

    # Check results
    assert isinstance(delay_m, (float, np.floating))

    # Typical ionospheric delays at L1 frequency fall between 0 and ~30 meters.
    # The exact value depends on the local time and geometry.
    assert delay_m >= 0.0
    assert delay_m < 50.0


def test_klobuchar_from_rinex_nav():
    """
    Test the initialization from RINEX NAV parsed dict.
    """
    header_data = {
        "ion_alpha": [1.4901e-08, 1.4901e-08, -5.9605e-08, -5.9605e-08],
        "ion_beta": [9.2160e04, 4.9152e04, -1.3107e05, -3.2768e05],
    }

    model = KlobucharModel.from_rinex_nav(header_data)

    assert np.allclose(model.alpha, header_data["ion_alpha"])
    assert np.allclose(model.beta, header_data["ion_beta"])


def test_klobuchar_negative_elevation():
    """
    Test that negative or zero elevation returns 0.0 delay.
    """
    alpha = [1.4901e-08, 1.4901e-08, -5.9605e-08, -5.9605e-08]
    beta = [9.2160e04, 4.9152e04, -1.3107e05, -3.2768e05]
    model = KlobucharModel(alpha=alpha, beta=beta)

    dt = datetime(2025, 12, 24, 22, 0, 0)
    receiver_llh = np.array([np.radians(35.68), np.radians(139.76), 50.0])

    # Below horizon
    elevation = np.radians(-5.0)
    azimuth = np.radians(45.0)

    delay_m = model.calculate_delay(dt, receiver_llh, azimuth, elevation)
    assert delay_m == 0.0
