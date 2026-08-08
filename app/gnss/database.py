"""Database module for storing GNSS observations, satellite positions, and SPP solutions.

This module provides SQLite database support for storing:
- EpochObservations: Raw GNSS observations for each epoch
- Satellite positions: Computed satellite positions
- SPP solutions: Single Point Positioning results

The database schema is normalized to efficiently store observation data with
minimal redundancy while maintaining query performance.
"""

import datetime as dt
import json
from pathlib import Path

from sqlalchemy import (
    ForeignKey,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.gnss.satellite_signals import (
    AmbiguityObservation,
    EpochObservations,
    SatelliteObservation,
    SatelliteSignalObservation,
)


class Base(DeclarativeBase):
    pass


class Epoch(Base):
    """Observation epoch table.

    Each row represents a unique observation epoch (timestamp).
    """

    __tablename__ = "epochs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    datetime: Mapped[dt.datetime] = mapped_column(unique=True, index=True)

    # Relationships
    satellites: Mapped[list["Satellite"]] = relationship(back_populates="epoch", cascade="all, delete-orphan")
    spp_solution: Mapped["SppSolution | None"] = relationship(
        back_populates="epoch",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Epoch(id={self.id}, datetime={self.datetime})>"


class Satellite(Base):
    """Satellite observations for a specific epoch.

    Each row represents observations for a single satellite at one epoch.
    """

    __tablename__ = "satellites"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    epoch_id: Mapped[int] = mapped_column(ForeignKey("epochs.id"), index=True)
    satellite_id: Mapped[str] = mapped_column(String(10), index=True)  # e.g., "G01", "E05", "R10"
    prn: Mapped[int] = mapped_column()
    system: Mapped[str] = mapped_column(String(10))  # "GPS", "QZSS", "Galileo", "GLONASS"

    # Relationships
    epoch: Mapped["Epoch"] = relationship(back_populates="satellites")
    signals: Mapped[list["Signal"]] = relationship(back_populates="satellite", cascade="all, delete-orphan")
    ambiguities: Mapped[list["Ambiguity"]] = relationship(back_populates="satellite", cascade="all, delete-orphan")
    position: Mapped["SatellitePosition | None"] = relationship(
        back_populates="satellite",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<Satellite(id={self.id}, satellite_id={self.satellite_id}, epoch_id={self.epoch_id})>"


class Signal(Base):
    """Signal observation for a specific satellite and band.

    Each row represents signal measurements (pseudorange, carrier phase, etc.)
    for a specific frequency band.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    satellite_id: Mapped[int] = mapped_column(ForeignKey("satellites.id"), index=True)
    band: Mapped[str] = mapped_column(String(10))  # e.g., "L1", "L2", "L5", "E1", "E5a"
    pseudorange: Mapped[float] = mapped_column()  # in meters
    carrier_phase: Mapped[float] = mapped_column()  # in cycles
    doppler: Mapped[float] = mapped_column()  # in Hz
    snr: Mapped[float] = mapped_column()  # in dB-Hz

    # Relationships
    satellite: Mapped["Satellite"] = relationship(back_populates="signals")

    def __repr__(self):
        return f"<Signal(id={self.id}, band={self.band}, satellite_id={self.satellite_id})>"


class Ambiguity(Base):
    """Ambiguity observations for dual-frequency combinations.

    Each row represents ambiguity measurements for a specific frequency combination.
    """

    __tablename__ = "ambiguities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    satellite_id: Mapped[int] = mapped_column(ForeignKey("satellites.id"), index=True)
    combination: Mapped[str] = mapped_column(String(20))  # e.g., "L1_L2", "L1_L5"
    widelane: Mapped[float] = mapped_column()  # in cycles
    ionofree: Mapped[float] = mapped_column()  # in cycles
    geofree: Mapped[float | None] = mapped_column(default=None)  # in cycles
    multipath: Mapped[float | None] = mapped_column(default=None)  # in meters

    # Relationships
    satellite: Mapped["Satellite"] = relationship(back_populates="ambiguities")

    def __repr__(self):
        return f"<Ambiguity(id={self.id}, combination={self.combination}, satellite_id={self.satellite_id})>"


class SatellitePosition(Base):
    """Computed satellite position in ECEF coordinates.

    Each row represents the computed position of a satellite at a specific epoch.
    """

    __tablename__ = "satellite_positions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    satellite_id: Mapped[int] = mapped_column(ForeignKey("satellites.id"), unique=True, index=True)
    datetime: Mapped[dt.datetime] = mapped_column(index=True)  # Observation time (UTC)
    nano_second: Mapped[int] = mapped_column(default=0)  # Sub-second value in nanoseconds
    x: Mapped[float] = mapped_column()  # ECEF X coordinate in meters
    y: Mapped[float] = mapped_column()  # ECEF Y coordinate in meters
    z: Mapped[float] = mapped_column()  # ECEF Z coordinate in meters
    clock_bias: Mapped[float | None] = mapped_column(default=None)  # Satellite clock bias in seconds

    # Relationships
    satellite: Mapped["Satellite"] = relationship(back_populates="position")

    def __repr__(self):
        return f"<SatellitePosition(id={self.id}, satellite_id={self.satellite_id})>"


class SppSolution(Base):
    """Single Point Positioning solution for an epoch.

    Each row represents the computed receiver position and clock bias for one epoch.
    """

    __tablename__ = "spp_solutions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    epoch_id: Mapped[int] = mapped_column(ForeignKey("epochs.id"), unique=True, index=True)

    # ECEF coordinates
    x: Mapped[float] = mapped_column()  # ECEF X coordinate in meters
    y: Mapped[float] = mapped_column()  # ECEF Y coordinate in meters
    z: Mapped[float] = mapped_column()  # ECEF Z coordinate in meters

    # LLH coordinates
    latitude: Mapped[float] = mapped_column()  # in degrees
    longitude: Mapped[float] = mapped_column()  # in degrees
    height: Mapped[float] = mapped_column()  # in meters

    # Clock bias
    clock_bias_m: Mapped[float] = mapped_column()  # in meters

    # Solution quality
    num_satellites: Mapped[int] = mapped_column()
    residuals: Mapped[str | None] = mapped_column(Text, default=None)  # JSON array of residuals

    # Relationships
    epoch: Mapped["Epoch"] = relationship(back_populates="spp_solution")

    def __repr__(self):
        return f"<SppSolution(id={self.id}, epoch_id={self.epoch_id}, num_satellites={self.num_satellites})>"


class GnssDatabase:
    """Database manager for GNSS observations and solutions.

    This class provides high-level functions to interact with the SQLite database
    for storing and retrieving GNSS observation data.

    Example:
        >>> db = GnssDatabase("gnss_data.db")
        >>> db.save_epoch_observations(epoch_observations_list)
        >>> loaded_epochs = db.load_epoch_observations()
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            # Enable foreign key support in SQLite
            connect_args={"check_same_thread": False},
            echo=False,
        )

        # Enable foreign key constraints for SQLite
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_epoch_observations(self, observations: list[EpochObservations], session: Session | None = None) -> None:
        """Save a list of EpochObservations to the database.

        Args:
            observations: List of EpochObservations to save
            session: Optional existing session (if None, creates new one)
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            for epoch_obs in observations:
                self._save_single_epoch(session, epoch_obs)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if close_session:
                session.close()

    def _save_single_epoch(self, session: Session, epoch_obs: EpochObservations) -> Epoch:
        """Save a single EpochObservations to the database.

        Args:
            session: Database session
            epoch_obs: EpochObservations to save

        Returns:
            Created Epoch object
        """
        # Check if epoch already exists
        existing_epoch = session.query(Epoch).filter_by(datetime=epoch_obs.datetime).first()
        if existing_epoch:
            # Delete existing epoch (cascade will delete related data)
            session.delete(existing_epoch)
            session.flush()

        # Create new epoch
        epoch = Epoch(datetime=epoch_obs.datetime)
        session.add(epoch)
        session.flush()  # Get epoch.id

        # Save all satellites
        for sat_id, sat_obs in epoch_obs.iter_satellites():
            self._save_satellite(session, epoch, sat_id, sat_obs)

        return epoch

    def _save_satellite(self, session: Session, epoch: Epoch, sat_id: str, sat_obs: SatelliteObservation) -> Satellite:
        """Save a single SatelliteObservation to the database.

        Args:
            session: Database session
            epoch: Parent Epoch object
            sat_id: Satellite ID (e.g., "G01")
            sat_obs: SatelliteObservation to save

        Returns:
            Created Satellite object
        """
        # Determine system from satellite ID
        system_map = {
            "G": "GPS",
            "J": "QZSS",
            "E": "Galileo",
            "R": "GLONASS",
        }
        system = system_map.get(sat_id[0], "Unknown")

        satellite = Satellite(
            epoch_id=epoch.id,
            satellite_id=sat_id,
            prn=sat_obs.prn,
            system=system,
        )
        session.add(satellite)
        session.flush()  # Get satellite.id

        # Save signals
        for band, signal_obs in sat_obs.signals.items():
            signal = Signal(
                satellite_id=satellite.id,
                band=band,
                pseudorange=signal_obs.pseudorange,
                carrier_phase=signal_obs.carrier_phase,
                doppler=signal_obs.doppler_,
                snr=signal_obs.snr,
            )
            session.add(signal)

        # Save ambiguities
        for combination, amb_obs in sat_obs.ambiguities.items():
            ambiguity = Ambiguity(
                satellite_id=satellite.id,
                combination=combination,
                widelane=amb_obs.widelane,
                ionofree=amb_obs.ionofree,
                geofree=amb_obs.geofree,
                multipath=amb_obs.multipath,
            )
            session.add(ambiguity)

        return satellite

    def load_epoch_observations(
        self,
        start_datetime: dt.datetime | None = None,
        end_datetime: dt.datetime | None = None,
        session: Session | None = None,
    ) -> list[EpochObservations]:
        """Load EpochObservations from the database.

        Args:
            start_datetime: Optional start time filter
            end_datetime: Optional end time filter
            session: Optional existing session (if None, creates new one)

        Returns:
            List of EpochObservations
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            query = session.query(Epoch)

            if start_datetime:
                query = query.filter(Epoch.datetime >= start_datetime)
            if end_datetime:
                query = query.filter(Epoch.datetime <= end_datetime)

            query = query.order_by(Epoch.datetime)
            epochs = query.all()

            result = []
            for epoch in epochs:
                epoch_obs = self._load_single_epoch(session, epoch)
                result.append(epoch_obs)

            return result
        finally:
            if close_session:
                session.close()

    def _load_single_epoch(self, session: Session, epoch: Epoch) -> EpochObservations:
        """Load a single EpochObservations from the database.

        Args:
            session: Database session
            epoch: Epoch object to load

        Returns:
            EpochObservations object
        """
        satellites_gps = []
        satellites_qzss = []
        satellites_galileo = []
        satellites_glonass = []

        for satellite in epoch.satellites:
            sat_obs = self._load_satellite(session, satellite)

            if satellite.system == "GPS":
                satellites_gps.append(sat_obs)
            elif satellite.system == "QZSS":
                satellites_qzss.append(sat_obs)
            elif satellite.system == "Galileo":
                satellites_galileo.append(sat_obs)
            elif satellite.system == "GLONASS":
                satellites_glonass.append(sat_obs)

        return EpochObservations(
            datetime=epoch.datetime,
            satellites_gps=satellites_gps,
            satellites_qzss=satellites_qzss,
            satellites_galileo=satellites_galileo,
            satellites_glonass=satellites_glonass,
        )

    def _load_satellite(self, session: Session, satellite: Satellite) -> SatelliteObservation:
        """Load a single SatelliteObservation from the database.

        Args:
            session: Database session
            satellite: Satellite object to load

        Returns:
            SatelliteObservation object
        """
        signals = {}
        for signal in satellite.signals:
            signals[signal.band] = SatelliteSignalObservation(
                pseudorange=signal.pseudorange,
                carrier_phase=signal.carrier_phase,
                doppler_=signal.doppler,
                snr=signal.snr,
            )

        ambiguities = {}
        for ambiguity in satellite.ambiguities:
            ambiguities[ambiguity.combination] = AmbiguityObservation(
                widelane=ambiguity.widelane,
                ionofree=ambiguity.ionofree,
                geofree=ambiguity.geofree if ambiguity.geofree is not None else 0.0,
                multipath=ambiguity.multipath if ambiguity.multipath is not None else 0.0,
            )

        return SatelliteObservation(
            prn=satellite.prn,
            signals=signals,
            ambiguities=ambiguities,
        )

    def save_satellite_positions(
        self,
        positions: dict[str, dict],
        epoch_datetime: dt.datetime,
        session: Session | None = None,
    ) -> None:
        """Save satellite positions for a specific epoch.

        Args:
            positions: Dictionary mapping satellite_id to position dict with keys:
                       'x', 'y', 'z', 'clock_bias' (optional),
                       'datetime' (datetime), 'nano_second' (int, optional)
            epoch_datetime: Datetime of the epoch
            session: Optional existing session (if None, creates new one)
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            epoch = session.query(Epoch).filter_by(datetime=epoch_datetime).first()
            if not epoch:
                raise ValueError(f"Epoch not found: {epoch_datetime}")

            for sat_id, pos_data in positions.items():
                satellite = session.query(Satellite).filter_by(epoch_id=epoch.id, satellite_id=sat_id).first()

                if not satellite:
                    continue  # Skip if satellite not found

                # Check if position already exists
                existing_pos = session.query(SatellitePosition).filter_by(satellite_id=satellite.id).first()

                if existing_pos:
                    # Update existing position
                    existing_pos.datetime = pos_data.get("datetime", epoch_datetime)
                    existing_pos.nano_second = pos_data.get("nano_second", 0)
                    existing_pos.x = pos_data["x"]
                    existing_pos.y = pos_data["y"]
                    existing_pos.z = pos_data["z"]
                    existing_pos.clock_bias = pos_data.get("clock_bias")
                else:
                    # Create new position
                    sat_pos = SatellitePosition(
                        satellite_id=satellite.id,
                        datetime=pos_data.get("datetime", epoch_datetime),
                        nano_second=pos_data.get("nano_second", 0),
                        x=pos_data["x"],
                        y=pos_data["y"],
                        z=pos_data["z"],
                        clock_bias=pos_data.get("clock_bias"),
                    )
                    session.add(sat_pos)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if close_session:
                session.close()

    def save_spp_solution(
        self,
        solution_data: dict,
        epoch_datetime: dt.datetime,
        session: Session | None = None,
    ) -> None:
        """Save SPP solution for a specific epoch.

        Args:
            solution_data: Dictionary with keys:
                          'position_ecef': [x, y, z]
                          'position_llh': [lat, lon, height]
                          'clock_bias_m': float
                          'num_satellites': int
                          'residuals': list of floats (optional)
            epoch_datetime: Datetime of the epoch
            session: Optional existing session (if None, creates new one)
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            epoch = session.query(Epoch).filter_by(datetime=epoch_datetime).first()
            if not epoch:
                raise ValueError(f"Epoch not found: {epoch_datetime}")

            # Check if solution already exists
            existing_sol = session.query(SppSolution).filter_by(epoch_id=epoch.id).first()

            residuals_json = None
            if "residuals" in solution_data and solution_data["residuals"] is not None:
                # Convert numpy array to list if necessary
                residuals = solution_data["residuals"]
                if hasattr(residuals, "tolist"):
                    residuals = residuals.tolist()
                residuals_json = json.dumps(residuals)

            if existing_sol:
                # Update existing solution
                existing_sol.x = solution_data["position_ecef"][0]
                existing_sol.y = solution_data["position_ecef"][1]
                existing_sol.z = solution_data["position_ecef"][2]
                existing_sol.latitude = solution_data["position_llh"][0]
                existing_sol.longitude = solution_data["position_llh"][1]
                existing_sol.height = solution_data["position_llh"][2]
                existing_sol.clock_bias_m = solution_data["clock_bias_m"]
                existing_sol.num_satellites = solution_data["num_satellites"]
                existing_sol.residuals = residuals_json
            else:
                # Create new solution
                spp_sol = SppSolution(
                    epoch_id=epoch.id,
                    x=solution_data["position_ecef"][0],
                    y=solution_data["position_ecef"][1],
                    z=solution_data["position_ecef"][2],
                    latitude=solution_data["position_llh"][0],
                    longitude=solution_data["position_llh"][1],
                    height=solution_data["position_llh"][2],
                    clock_bias_m=solution_data["clock_bias_m"],
                    num_satellites=solution_data["num_satellites"],
                    residuals=residuals_json,
                )
                session.add(spp_sol)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if close_session:
                session.close()

    def get_statistics(self, session: Session | None = None) -> dict:
        """Get database statistics.

        Args:
            session: Optional existing session (if None, creates new one)

        Returns:
            Dictionary with statistics
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            stats: dict[str, object] = {
                "num_epochs": session.query(Epoch).count(),
                "num_satellites": session.query(Satellite).count(),
                "num_signals": session.query(Signal).count(),
                "num_ambiguities": session.query(Ambiguity).count(),
                "num_satellite_positions": session.query(SatellitePosition).count(),
                "num_spp_solutions": session.query(SppSolution).count(),
            }

            # Get time range
            first_epoch = session.query(Epoch).order_by(Epoch.datetime).first()
            last_epoch = session.query(Epoch).order_by(Epoch.datetime.desc()).first()

            if first_epoch and last_epoch:
                stats["first_epoch"] = first_epoch.datetime
                stats["last_epoch"] = last_epoch.datetime
                stats["time_span"] = (last_epoch.datetime - first_epoch.datetime).total_seconds()

            return stats
        finally:
            if close_session:
                session.close()

    def clear_database(self, session: Session | None = None) -> None:
        """Clear all data from the database.

        Args:
            session: Optional existing session (if None, creates new one)
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True

        try:
            # Delete all epochs - cascading will handle related tables
            # We need to load and delete objects for cascade to work
            epochs = session.query(Epoch).all()
            for epoch in epochs:
                session.delete(epoch)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            if close_session:
                session.close()
