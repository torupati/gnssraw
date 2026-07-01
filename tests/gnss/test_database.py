"""Tests for GNSS database module."""

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import pytest
import numpy as np

from app.gnss.database import GnssDatabase
from app.gnss.satellite_signals import (
    EpochObservations,
    SatelliteObservation,
    SatelliteSignalObservation,
    AmbiguityObservation,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_gnss.db"
        db = GnssDatabase(db_path)
        yield db
        # Cleanup happens automatically


@pytest.fixture
def sample_epoch_observations():
    """Create sample EpochObservations for testing."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    # GPS satellite
    gps_sat = SatelliteObservation(
        prn=1,
        signals={
            "L1": SatelliteSignalObservation(
                pseudorange=20000000.0,
                carrier_phase=105263157.9,
                doppler_=-1234.5,
                snr=45.0,
            ),
            "L2": SatelliteSignalObservation(
                pseudorange=20000010.0,
                carrier_phase=82021028.7,
                doppler_=-962.3,
                snr=42.0,
            ),
        },
        ambiguities={
            "L1_L2": AmbiguityObservation(
                widelane=123.45,
                ionofree=234.56,
                geofree=1.23,
                multipath=0.5,
            ),
        },
    )

    # Galileo satellite
    galileo_sat = SatelliteObservation(
        prn=5,
        signals={
            "E1": SatelliteSignalObservation(
                pseudorange=21000000.0,
                carrier_phase=110526315.8,
                doppler_=-2000.0,
                snr=48.0,
            ),
            "E5a": SatelliteSignalObservation(
                pseudorange=21000015.0,
                carrier_phase=99487471.2,
                doppler_=-1753.5,
                snr=46.0,
            ),
        },
        ambiguities={
            "E1_E5a": AmbiguityObservation(
                widelane=456.78,
                ionofree=567.89,
                geofree=2.34,
                multipath=0.7,
            ),
        },
    )

    epoch_obs = EpochObservations(
        datetime=dt,
        satellites_gps=[gps_sat],
        satellites_qzss=[],
        satellites_galileo=[galileo_sat],
        satellites_glonass=[],
    )

    return epoch_obs


def test_database_creation(temp_db):
    """Test that database is created successfully."""
    assert temp_db.db_path.exists()
    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 0
    assert stats["num_satellites"] == 0


def test_save_single_epoch(temp_db, sample_epoch_observations):
    """Test saving a single epoch of observations."""
    temp_db.save_epoch_observations([sample_epoch_observations])

    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 1
    assert stats["num_satellites"] == 2  # 1 GPS + 1 Galileo
    assert stats["num_signals"] == 4  # 2 per satellite
    assert stats["num_ambiguities"] == 2  # 1 per satellite


def test_save_and_load_epoch(temp_db, sample_epoch_observations):
    """Test saving and loading epoch observations."""
    temp_db.save_epoch_observations([sample_epoch_observations])

    loaded_epochs = temp_db.load_epoch_observations()
    assert len(loaded_epochs) == 1

    loaded_epoch = loaded_epochs[0]
    # SQLite doesn't preserve timezone info, so compare without timezone
    assert (
        loaded_epoch.datetime.replace(tzinfo=timezone.utc)
        == sample_epoch_observations.datetime
    )
    assert len(loaded_epoch.satellites_gps) == 1
    assert len(loaded_epoch.satellites_galileo) == 1

    # Check GPS satellite
    gps_sat = loaded_epoch.satellites_gps[0]
    assert gps_sat.prn == 1
    assert "L1" in gps_sat.signals
    assert "L2" in gps_sat.signals
    assert gps_sat.signals["L1"].pseudorange == 20000000.0
    assert gps_sat.signals["L1"].carrier_phase == 105263157.9
    assert gps_sat.signals["L1"].doppler_ == -1234.5
    assert gps_sat.signals["L1"].snr == 45.0

    # Check ambiguities
    assert "L1_L2" in gps_sat.ambiguities
    assert gps_sat.ambiguities["L1_L2"].widelane == 123.45
    assert gps_sat.ambiguities["L1_L2"].ionofree == 234.56

    # Check Galileo satellite
    galileo_sat = loaded_epoch.satellites_galileo[0]
    assert galileo_sat.prn == 5
    assert "E1" in galileo_sat.signals
    assert "E5a" in galileo_sat.signals


def test_save_multiple_epochs(temp_db):
    """Test saving multiple epochs."""
    epochs = []
    for i in range(5):
        dt = datetime(2024, 1, 15, 12, 0, i, tzinfo=timezone.utc)
        sat = SatelliteObservation(
            prn=1,
            signals={
                "L1": SatelliteSignalObservation(
                    pseudorange=20000000.0 + i * 100,
                    carrier_phase=105263157.9 + i * 1000,
                    doppler_=-1234.5,
                    snr=45.0,
                ),
            },
            ambiguities={},
        )
        epoch_obs = EpochObservations(
            datetime=dt,
            satellites_gps=[sat],
            satellites_qzss=[],
            satellites_galileo=[],
            satellites_glonass=[],
        )
        epochs.append(epoch_obs)

    temp_db.save_epoch_observations(epochs)

    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 5
    assert stats["num_satellites"] == 5

    # Load and verify
    loaded_epochs = temp_db.load_epoch_observations()
    assert len(loaded_epochs) == 5

    # Check that epochs are ordered by time
    for i in range(4):
        assert loaded_epochs[i].datetime < loaded_epochs[i + 1].datetime


def test_load_with_time_filter(temp_db):
    """Test loading epochs with time filters."""
    epochs = []
    for i in range(10):
        dt = datetime(2024, 1, 15, 12, 0, i, tzinfo=timezone.utc)
        sat = SatelliteObservation(prn=1, signals={}, ambiguities={})
        epoch_obs = EpochObservations(
            datetime=dt,
            satellites_gps=[sat],
            satellites_qzss=[],
            satellites_galileo=[],
            satellites_glonass=[],
        )
        epochs.append(epoch_obs)

    temp_db.save_epoch_observations(epochs)

    # Load with start time filter
    start_dt = datetime(2024, 1, 15, 12, 0, 5, tzinfo=timezone.utc)
    loaded = temp_db.load_epoch_observations(start_datetime=start_dt)
    assert len(loaded) == 5

    # Load with end time filter
    end_dt = datetime(2024, 1, 15, 12, 0, 3, tzinfo=timezone.utc)
    loaded = temp_db.load_epoch_observations(end_datetime=end_dt)
    assert len(loaded) == 4

    # Load with both filters
    start_dt = datetime(2024, 1, 15, 12, 0, 2, tzinfo=timezone.utc)
    end_dt = datetime(2024, 1, 15, 12, 0, 7, tzinfo=timezone.utc)
    loaded = temp_db.load_epoch_observations(
        start_datetime=start_dt, end_datetime=end_dt
    )
    assert len(loaded) == 6


def test_save_satellite_positions(temp_db, sample_epoch_observations):
    """Test saving satellite positions."""
    temp_db.save_epoch_observations([sample_epoch_observations])

    positions = {
        "G01": {
            "datetime": sample_epoch_observations.datetime,
            "nano_second": 500000000,
            "x": 1234567.89,
            "y": 2345678.90,
            "z": 3456789.01,
            "clock_bias": 0.000123,
        },
        "E05": {
            "datetime": sample_epoch_observations.datetime,
            "nano_second": 250000000,
            "x": 9876543.21,
            "y": 8765432.10,
            "z": 7654321.09,
            "clock_bias": -0.000456,
        },
    }

    temp_db.save_satellite_positions(positions, sample_epoch_observations.datetime)

    stats = temp_db.get_statistics()
    assert stats["num_satellite_positions"] == 2


def test_save_qzss_satellite_position(temp_db):
    """Test persisting satellite position for a QZSS satellite."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    qzss_sat = SatelliteObservation(
        prn=193,
        signals={
            "L1": SatelliteSignalObservation(
                pseudorange=22000000.0,
                carrier_phase=115000000.0,
                doppler_=-1500.0,
                snr=44.0,
            )
        },
        ambiguities={},
    )
    epoch_obs = EpochObservations(
        datetime=dt,
        satellites_gps=[],
        satellites_qzss=[qzss_sat],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    temp_db.save_epoch_observations([epoch_obs])

    positions = {
        "J193": {
            "datetime": dt,
            "nano_second": 0,
            "x": 1234.0,
            "y": 5678.0,
            "z": 91011.0,
            "clock_bias": 1.0e-4,
        }
    }
    temp_db.save_satellite_positions(positions, dt)

    stats = temp_db.get_statistics()
    assert stats["num_satellite_positions"] == 1


def test_save_spp_solution(temp_db, sample_epoch_observations):
    """Test saving SPP solution."""
    temp_db.save_epoch_observations([sample_epoch_observations])

    solution_data = {
        "position_ecef": np.array([-3962108.673, 3381309.574, 3668678.638]),
        "position_llh": np.array([35.7105, 139.8107, 45.3]),
        "clock_bias_m": 12345.678,
        "num_satellites": 8,
        "residuals": np.array([0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4, -0.1]),
    }

    temp_db.save_spp_solution(solution_data, sample_epoch_observations.datetime)

    stats = temp_db.get_statistics()
    assert stats["num_spp_solutions"] == 1


def test_update_existing_epoch(temp_db, sample_epoch_observations):
    """Test that saving an epoch with the same datetime updates the existing one."""
    # Save first version
    temp_db.save_epoch_observations([sample_epoch_observations])
    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 1
    assert stats["num_satellites"] == 2

    # Create a modified version with different data
    modified_epoch = EpochObservations(
        datetime=sample_epoch_observations.datetime,
        satellites_gps=[
            SatelliteObservation(
                prn=2,
                signals={
                    "L1": SatelliteSignalObservation(
                        pseudorange=30000000.0,
                        carrier_phase=157894736.8,
                        doppler_=-5000.0,
                        snr=50.0,
                    ),
                },
                ambiguities={},
            )
        ],
        satellites_qzss=[],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    # Save modified version
    temp_db.save_epoch_observations([modified_epoch])

    # Should still have only 1 epoch, but with updated data
    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 1
    assert stats["num_satellites"] == 1  # Only 1 GPS satellite now

    # Verify the data is updated
    loaded_epochs = temp_db.load_epoch_observations()
    assert len(loaded_epochs) == 1
    assert len(loaded_epochs[0].satellites_gps) == 1
    assert loaded_epochs[0].satellites_gps[0].prn == 2


def test_database_statistics(temp_db):
    """Test getting database statistics."""
    # Empty database
    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 0
    assert "first_epoch" not in stats
    assert "last_epoch" not in stats

    # Add some data
    epochs = []
    for i in range(3):
        dt = datetime(2024, 1, 15, 12, 0, i, tzinfo=timezone.utc)
        sat = SatelliteObservation(
            prn=1,
            signals={
                "L1": SatelliteSignalObservation(
                    pseudorange=20000000.0,
                    carrier_phase=105263157.9,
                    doppler_=-1234.5,
                    snr=45.0,
                ),
            },
            ambiguities={
                "L1_L2": AmbiguityObservation(
                    widelane=123.45,
                    ionofree=234.56,
                ),
            },
        )
        epoch_obs = EpochObservations(
            datetime=dt,
            satellites_gps=[sat],
            satellites_qzss=[],
            satellites_galileo=[],
            satellites_glonass=[],
        )
        epochs.append(epoch_obs)

    temp_db.save_epoch_observations(epochs)

    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 3
    assert stats["num_satellites"] == 3
    assert stats["num_signals"] == 3
    assert stats["num_ambiguities"] == 3
    assert "first_epoch" in stats
    assert "last_epoch" in stats
    assert stats["time_span"] == 2.0  # 2 seconds


def test_clear_database(temp_db, sample_epoch_observations):
    """Test clearing all data from database."""
    temp_db.save_epoch_observations([sample_epoch_observations])
    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 1

    temp_db.clear_database()

    stats = temp_db.get_statistics()
    assert stats["num_epochs"] == 0
    assert stats["num_satellites"] == 0
    assert stats["num_signals"] == 0


def test_empty_satellite_lists(temp_db):
    """Test handling epochs with empty satellite lists."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    epoch_obs = EpochObservations(
        datetime=dt,
        satellites_gps=[],
        satellites_qzss=[],
        satellites_galileo=[],
        satellites_glonass=[],
    )

    temp_db.save_epoch_observations([epoch_obs])

    loaded_epochs = temp_db.load_epoch_observations()
    assert len(loaded_epochs) == 1
    assert len(loaded_epochs[0].satellites_gps) == 0
    assert len(loaded_epochs[0].satellites_galileo) == 0


def test_multiple_constellations(temp_db):
    """Test handling multiple GNSS constellations."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

    gps_sat = SatelliteObservation(prn=1, signals={}, ambiguities={})
    qzss_sat = SatelliteObservation(prn=193, signals={}, ambiguities={})
    galileo_sat = SatelliteObservation(prn=5, signals={}, ambiguities={})
    glonass_sat = SatelliteObservation(prn=1, signals={}, ambiguities={})

    epoch_obs = EpochObservations(
        datetime=dt,
        satellites_gps=[gps_sat],
        satellites_qzss=[qzss_sat],
        satellites_galileo=[galileo_sat],
        satellites_glonass=[glonass_sat],
    )

    temp_db.save_epoch_observations([epoch_obs])

    loaded_epochs = temp_db.load_epoch_observations()
    assert len(loaded_epochs) == 1
    loaded = loaded_epochs[0]
    assert len(loaded.satellites_gps) == 1
    assert len(loaded.satellites_qzss) == 1
    assert len(loaded.satellites_galileo) == 1
    assert len(loaded.satellites_glonass) == 1


def test_spp_solution_without_residuals(temp_db, sample_epoch_observations):
    """Test saving SPP solution without residuals."""
    temp_db.save_epoch_observations([sample_epoch_observations])

    solution_data = {
        "position_ecef": np.array([-3962108.673, 3381309.574, 3668678.638]),
        "position_llh": np.array([35.7105, 139.8107, 45.3]),
        "clock_bias_m": 12345.678,
        "num_satellites": 8,
    }

    temp_db.save_spp_solution(solution_data, sample_epoch_observations.datetime)

    stats = temp_db.get_statistics()
    assert stats["num_spp_solutions"] == 1


def test_save_satellite_positions_missing_epoch(temp_db):
    """Test that saving positions for non-existent epoch raises error."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    positions = {"G01": {"x": 1234567.89, "y": 2345678.90, "z": 3456789.01}}

    with pytest.raises(ValueError, match="Epoch not found"):
        temp_db.save_satellite_positions(positions, dt)


def test_save_spp_solution_missing_epoch(temp_db):
    """Test that saving SPP solution for non-existent epoch raises error."""
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    solution_data = {
        "position_ecef": np.array([1.0, 2.0, 3.0]),
        "position_llh": np.array([35.0, 139.0, 100.0]),
        "clock_bias_m": 0.0,
        "num_satellites": 4,
    }

    with pytest.raises(ValueError, match="Epoch not found"):
        temp_db.save_spp_solution(solution_data, dt)
