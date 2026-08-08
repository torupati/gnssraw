"""Tests for ambiguity statistics computation."""

import csv
from datetime import datetime
from pathlib import Path
import tempfile

from app.gnss.satellite_signals import (
    EpochObservations,
    SatelliteObservation,
    SatelliteSignalObservation,
    AmbiguityObservation,
    compute_ambiguity_statistics,
    save_ambiguity_statistics_to_csv,
)


def test_compute_ambiguity_statistics():
    """Test computing ambiguity statistics from epochs."""
    # Create test data with two epochs and one GPS satellite
    signal_l1 = SatelliteSignalObservation(
        pseudorange=20000000.0,
        carrier_phase=105000000.0,
        doppler_=100.0,
        snr=45.0,
    )
    signal_l2 = SatelliteSignalObservation(
        pseudorange=20000100.0,
        carrier_phase=81500000.0,
        doppler_=120.0,
        snr=42.0,
    )

    sat_obs1 = SatelliteObservation(
        prn=1,
        signals={"L1": signal_l1, "L2": signal_l2},
        ambiguities={
            "L1_L2": AmbiguityObservation(
                widelane=10.5, ionofree=100.0, geofree=5.0, multipath=0.5
            )
        },
    )

    epoch1 = EpochObservations(
        datetime=datetime(2025, 1, 1, 0, 0, 0),
        satellites_gps=[sat_obs1],
        satellites_qzss=[],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    # Second epoch with slightly different values
    signal_l1_2 = SatelliteSignalObservation(
        pseudorange=20000010.0,
        carrier_phase=105000050.0,
        doppler_=101.0,
        snr=46.0,
    )
    signal_l2_2 = SatelliteSignalObservation(
        pseudorange=20000110.0,
        carrier_phase=81500040.0,
        doppler_=121.0,
        snr=43.0,
    )

    sat_obs2 = SatelliteObservation(
        prn=1,
        signals={"L1": signal_l1_2, "L2": signal_l2_2},
        ambiguities={
            "L1_L2": AmbiguityObservation(
                widelane=10.7, ionofree=101.0, geofree=5.2, multipath=0.6
            )
        },
    )

    epoch2 = EpochObservations(
        datetime=datetime(2025, 1, 1, 0, 0, 1),
        satellites_gps=[sat_obs2],
        satellites_qzss=[],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    epochs = [epoch1, epoch2]

    # Compute statistics
    stats = compute_ambiguity_statistics(epochs)

    # Verify results
    assert len(stats) == 1, "Should have one satellite-band combination"

    stat = stats[0]
    assert stat["satellite"] == "G01", "Satellite ID should be G01"
    assert stat["band"] == "L1_L2", "Band should be L1_L2"
    assert stat["num_epochs"] == 2, "Should have 2 epochs"

    # Check widelane statistics
    expected_wl_mean = (10.5 + 10.7) / 2
    assert abs(stat["widelane_ambiguity_mean"] - expected_wl_mean) < 0.001
    assert abs(stat["widelane_ambiguity_max_min"] - 0.2) < 0.001
    assert stat["widelane_ambiguity_std"] > 0

    # Check SNR statistics
    expected_snr_mean = (45.0 + 46.0) / 2
    assert abs(stat["snr_mean"] - expected_snr_mean) < 0.001
    assert stat["snr_max_min"] == 1.0


def test_save_ambiguity_statistics_to_csv():
    """Test saving ambiguity statistics to CSV file."""
    # Create test data
    signal_l1 = SatelliteSignalObservation(
        pseudorange=20000000.0,
        carrier_phase=105000000.0,
        doppler_=100.0,
        snr=45.0,
    )
    signal_l2 = SatelliteSignalObservation(
        pseudorange=20000100.0,
        carrier_phase=81500000.0,
        doppler_=120.0,
        snr=42.0,
    )

    sat_obs = SatelliteObservation(
        prn=5,
        signals={"L1": signal_l1, "L2": signal_l2},
        ambiguities={
            "L1_L2": AmbiguityObservation(
                widelane=10.5, ionofree=100.0, geofree=5.0, multipath=0.5
            )
        },
    )

    epoch = EpochObservations(
        datetime=datetime(2025, 1, 1, 0, 0, 0),
        satellites_gps=[sat_obs],
        satellites_qzss=[],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    epochs = [epoch]

    # Save to CSV
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_statistics.csv"
        save_ambiguity_statistics_to_csv(epochs, csv_path)

        # Verify CSV file was created
        assert csv_path.exists(), "CSV file should be created"

        # Read and verify content
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 1, "Should have one row"

            row = rows[0]
            assert row["satellite"] == "G05"
            assert row["band"] == "L1_L2"
            assert row["num_epochs"] == "1"
            assert float(row["widelane_ambiguity_mean"]) == 10.5
            assert float(row["snr_mean"]) == 45.0


def test_multiple_satellites():
    """Test computing statistics for multiple satellites and bands."""
    # GPS satellite G01 with L1_L2
    sat1 = SatelliteObservation(
        prn=1,
        signals={
            "L1": SatelliteSignalObservation(20000000.0, 105000000.0, 100.0, 45.0),
            "L2": SatelliteSignalObservation(20000100.0, 81500000.0, 120.0, 42.0),
        },
        ambiguities={"L1_L2": AmbiguityObservation(10.5, 100.0, 5.0, 0.5)},
    )

    # Galileo satellite E10 with L1_L5
    sat2 = SatelliteObservation(
        prn=10,
        signals={
            "L1": SatelliteSignalObservation(20000000.0, 105000000.0, 100.0, 50.0),
            "L5": SatelliteSignalObservation(20000200.0, 78000000.0, 140.0, 48.0),
        },
        ambiguities={"L1_L5": AmbiguityObservation(8.3, 95.0, 4.0, 0.3)},
    )

    epoch = EpochObservations(
        datetime=datetime(2025, 1, 1, 0, 0, 0),
        satellites_gps=[sat1],
        satellites_qzss=[],
        satellites_galileo=[sat2],
        satellites_glonass=[],
    )

    stats = compute_ambiguity_statistics([epoch])

    assert len(stats) == 2, "Should have two satellite-band combinations"

    # Check that we have both satellites
    sat_ids = {s["satellite"] for s in stats}
    assert "G01" in sat_ids
    assert "E10" in sat_ids

    # Check bands
    bands = {s["band"] for s in stats}
    assert "L1_L2" in bands
    assert "L1_L5" in bands
