"""
GPS Ephemeris handling module
Provides GPS ephemeris data structures and satellite position computation.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np

from app.gnss.constants import CLIGHT

# Physical constants
GM_WGS84 = 3.986005e14  # WGS84 Earth's gravitational constant [m^3/s^2]
OMEGA_E = 7.2921151467e-5  # Earth's rotation rate [rad/s]
F_REL = -4.442807633e-10  # Relativistic correction factor

# GPS time reference epoch (start of GPS time)
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)


def datetime_to_gps_week_seconds(dt: datetime) -> tuple[int, float]:
    """
    Convert a UTC datetime to GPS week number and seconds of week.

    Args:
        dt: datetime in UTC (timezone-aware or naive). Naive datetimes are
            treated as UTC.

    Returns:
        A tuple (gps_week, sow) where:
            gps_week: GPS week number since GPS_EPOCH.
            sow: seconds of week [0, 604800).
    """
    # Ensure we operate in UTC and with a naive datetime for arithmetic
    if dt.tzinfo is not None:
        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        dt_utc = dt

    delta = dt_utc - GPS_EPOCH
    total_seconds = delta.total_seconds()
    week = int(total_seconds // 604800.0)
    sow = total_seconds - week * 604800.0
    return week, sow


class GPSEphemeris:
    """GPS satellite ephemeris class"""

    def __init__(self):
        self.prn = None  # Satellite PRN number
        self.toc = None  # Clock correction reference time
        self.toe = None  # Ephemeris reference time
        self.week = None  # GPS week

        # Clock parameters
        self.af0 = 0.0  # Clock bias [s]
        self.af1 = 0.0  # Clock drift [s/s]
        self.af2 = 0.0  # Clock drift rate [s/s^2]

        # Orbital parameters
        self.sqrtA = 0.0  # Square root of semi-major axis [m^1/2]
        self.e = 0.0  # Eccentricity
        self.i0 = 0.0  # Inclination angle [rad]
        self.Omega0 = 0.0  # Right ascension of ascending node [rad]
        self.omega = 0.0  # Argument of perigee [rad]
        self.M0 = 0.0  # Mean anomaly [rad]
        self.delta_n = 0.0  # Mean motion correction [rad/s]
        self.Omegadot = 0.0  # Rate of right ascension [rad/s]
        self.idot = 0.0  # Rate of inclination angle [rad/s]

        # Perturbation correction parameters
        self.cuc = 0.0  # Cosine correction to argument of latitude [rad]
        self.cus = 0.0  # Sine correction to argument of latitude [rad]
        self.crc = 0.0  # Cosine correction to orbital radius [m]
        self.crs = 0.0  # Sine correction to orbital radius [m]
        self.cic = 0.0  # Cosine correction to inclination [rad]
        self.cis = 0.0  # Sine correction to inclination [rad]

        # Other parameters
        self.tgd = 0.0  # Group delay differential [s]
        self.iodc = 0.0  # Issue of data, clock
        self.iode = 0.0  # Issue of data, ephemeris

    def __str__(self):
        return f"GPSEphemeris(PRN={self.prn}, TOE={self.toe}, Week={self.week})"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        """
        Convert ephemeris data to dictionary format

        Returns:
            Dictionary containing all ephemeris parameters
        """
        return {
            "prn": self.prn,
            "toc": self.toc.isoformat() if self.toc else None,
            "toe": self.toe,
            "week": self.week,
            # Clock parameters
            "af0": self.af0,
            "af1": self.af1,
            "af2": self.af2,
            # Orbital parameters
            "sqrtA": self.sqrtA,
            "e": self.e,
            "i0": self.i0,
            "Omega0": self.Omega0,
            "omega": self.omega,
            "M0": self.M0,
            "delta_n": self.delta_n,
            "Omegadot": self.Omegadot,
            "idot": self.idot,
            # Perturbation correction parameters
            "cuc": self.cuc,
            "cus": self.cus,
            "crc": self.crc,
            "crs": self.crs,
            "cic": self.cic,
            "cis": self.cis,
            # Other parameters
            "tgd": self.tgd,
            "iodc": self.iodc,
            "iode": self.iode,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert ephemeris data to JSON string

        Args:
            indent: Number of spaces for indentation (None for compact format)

        Returns:
            JSON string representation of ephemeris data
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "GPSEphemeris":
        """
        Create GPSEphemeris instance from dictionary

        Args:
            data: Dictionary containing ephemeris parameters

        Returns:
            GPSEphemeris instance
        """
        eph = cls()
        eph.prn = data.get("prn")
        eph.toc = datetime.fromisoformat(data["toc"]) if data.get("toc") else None
        eph.toe = data.get("toe")
        eph.week = data.get("week")

        # Clock parameters
        eph.af0 = data.get("af0", 0.0)
        eph.af1 = data.get("af1", 0.0)
        eph.af2 = data.get("af2", 0.0)

        # Orbital parameters
        eph.sqrtA = data.get("sqrtA", 0.0)
        eph.e = data.get("e", 0.0)
        eph.i0 = data.get("i0", 0.0)
        eph.Omega0 = data.get("Omega0", 0.0)
        eph.omega = data.get("omega", 0.0)
        eph.M0 = data.get("M0", 0.0)
        eph.delta_n = data.get("delta_n", 0.0)
        eph.Omegadot = data.get("Omegadot", 0.0)
        eph.idot = data.get("idot", 0.0)

        # Perturbation correction parameters
        eph.cuc = data.get("cuc", 0.0)
        eph.cus = data.get("cus", 0.0)
        eph.crc = data.get("crc", 0.0)
        eph.crs = data.get("crs", 0.0)
        eph.cic = data.get("cic", 0.0)
        eph.cis = data.get("cis", 0.0)

        # Other parameters
        eph.tgd = data.get("tgd", 0.0)
        eph.iodc = data.get("iodc", 0.0)
        eph.iode = data.get("iode", 0.0)

        return eph

    @classmethod
    def from_json(cls, json_str: str) -> "GPSEphemeris":
        """
        Create GPSEphemeris instance from JSON string

        Args:
            json_str: JSON string containing ephemeris parameters

        Returns:
            GPSEphemeris instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


def read_rinex_nav(
    nav_file: str,
) -> tuple[Dict[str, List[GPSEphemeris]], Dict[str, list[float]]]:
    """
    Read RINEX navigation file

    Args:
        nav_file: Navigation file path

    Returns:
        Tuple of (ephemerides dict, ionosphere parameters dict).
    """
    ephemerides: Dict[str, List[GPSEphemeris]] = {}
    ion_params: Dict[str, list[float]] = {}

    with open(nav_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Read header
    i = 0
    while i < len(lines) and "END OF HEADER" not in lines[i]:
        line = lines[i]
        # Check for IONOSPHERIC CORR
        if "IONOSPHERIC CORR" in line:
            if line.startswith("GPSA") or "ION ALPHA" in line:
                try:
                    alpha0 = float(line[5:17].replace("D", "E"))
                    alpha1 = float(line[17:29].replace("D", "E"))
                    alpha2 = float(line[29:41].replace("D", "E"))
                    alpha3 = float(line[41:53].replace("D", "E"))
                    ion_params["ion_alpha"] = [alpha0, alpha1, alpha2, alpha3]
                except ValueError:
                    pass
            elif line.startswith("GPSB") or "ION BETA" in line:
                try:
                    beta0 = float(line[5:17].replace("D", "E"))
                    beta1 = float(line[17:29].replace("D", "E"))
                    beta2 = float(line[29:41].replace("D", "E"))
                    beta3 = float(line[41:53].replace("D", "E"))
                    ion_params["ion_beta"] = [beta0, beta1, beta2, beta3]
                except ValueError:
                    pass
        i += 1
    i += 1

    # Read ephemeris data
    while i < len(lines):
        line = lines[i]

        # Process GPS and QZSS satellite data
        if not (line.startswith("G") or line.startswith("J")):
            i += 1
            continue

        eph = GPSEphemeris()

        # Line 1: Satellite ID, time, clock correction parameters
        prn = int(line[1:3])
        eph.prn = prn
        year = int(line[4:8])
        month = int(line[9:11])
        day = int(line[12:14])
        hour = int(line[15:17])
        minute = int(line[18:20])
        second = float(line[21:23])

        eph.toc = datetime(year, month, day, hour, minute, int(second))
        eph.af0 = float(line[23:42].replace("D", "E"))
        eph.af1 = float(line[42:61].replace("D", "E"))
        eph.af2 = float(line[61:80].replace("D", "E"))

        # Line 2
        i += 1
        line = lines[i]
        eph.iode = float(line[4:23].replace("D", "E"))
        eph.crs = float(line[23:42].replace("D", "E"))
        eph.delta_n = float(line[42:61].replace("D", "E"))
        eph.M0 = float(line[61:80].replace("D", "E"))

        # Line 3
        i += 1
        line = lines[i]
        eph.cuc = float(line[4:23].replace("D", "E"))
        eph.e = float(line[23:42].replace("D", "E"))
        eph.cus = float(line[42:61].replace("D", "E"))
        eph.sqrtA = float(line[61:80].replace("D", "E"))

        # Line 4
        i += 1
        line = lines[i]
        eph.toe = float(line[4:23].replace("D", "E"))
        eph.cic = float(line[23:42].replace("D", "E"))
        eph.Omega0 = float(line[42:61].replace("D", "E"))
        eph.cis = float(line[61:80].replace("D", "E"))

        # Line 5
        i += 1
        line = lines[i]
        eph.i0 = float(line[4:23].replace("D", "E"))
        eph.crc = float(line[23:42].replace("D", "E"))
        eph.omega = float(line[42:61].replace("D", "E"))
        eph.Omegadot = float(line[61:80].replace("D", "E"))

        # Line 6
        i += 1
        line = lines[i]
        eph.idot = float(line[4:23].replace("D", "E"))

        # Line 7
        i += 1
        line = lines[i]
        eph.week = int(float(line[23:42].replace("D", "E")))

        # Line 8 (SV accuracy, SV health, TGD, IODC)
        i += 1
        if i < len(lines):
            line = lines[i]
            if len(line) > 61:
                tgd_str = line[42:61].strip()
                if tgd_str:
                    eph.tgd = float(tgd_str.replace("D", "E"))
            if len(line) > 80:
                iodc_str = line[61:80].strip()
                if iodc_str:
                    eph.iodc = float(iodc_str.replace("D", "E"))

        # Add ephemeris to list
        sat_id = f"G{prn:02d}"
        if sat_id not in ephemerides:
            ephemerides[sat_id] = []
        ephemerides[sat_id].append(eph)

        i += 1

    return ephemerides, ion_params


# -----------------------------------------
# Satellite position computation functions
# -----------------------------------------
def solve_kepler(M: float, e: float) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E using Newton's method"""
    E = M
    for _ in range(10):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        dE = -f / f_prime
        E += dE
        if abs(dE) < 1e-12:
            break
    return E


def _wrap_time_diff(dt: float) -> float:
    if dt > 302400.0:
        return dt - 604800.0
    if dt < -302400.0:
        return dt + 604800.0
    return dt


def broadcast_ecef_and_clock(
    nav: GPSEphemeris,
    sow: float,
) -> tuple[np.ndarray, float]:
    """
    Compute satellite ECEF position and clock bias from broadcast ephemeris.

    Evaluates the GPS broadcast ephemeris orbital parameters at the given
    time to produce the satellite position in WGS84 ECEF coordinates and
    the satellite clock correction (including relativistic effects).

    Args:
        nav: GPS broadcast ephemeris containing orbital and clock parameters.
        sow: Transmission time as GPS seconds of week.

    Returns:
        Tuple of:
        - Satellite position in ECEF coordinates [x, y, z] (meters).
        - Satellite clock bias (seconds), including relativistic correction.
    """
    tk = _wrap_time_diff(sow - nav.toe)
    A = nav.sqrtA**2
    n0 = np.sqrt(GM_WGS84 / A**3)
    n = n0 + nav.delta_n
    M = nav.M0 + n * tk

    E = solve_kepler(M, nav.e)
    # print("Eccentric anomaly E:", E)
    v = np.arctan2(np.sqrt(1 - nav.e**2) * np.sin(E), np.cos(E) - nav.e)
    phi = v + nav.omega

    sin2p = np.sin(2 * phi)
    cos2p = np.cos(2 * phi)

    u = phi + nav.cus * sin2p + nav.cuc * cos2p
    r = A * (1 - nav.e * np.cos(E)) + nav.crs * sin2p + nav.crc * cos2p
    i = nav.i0 + nav.idot * tk + nav.cis * sin2p + nav.cic * cos2p

    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)

    Omega = nav.Omega0 + (nav.Omegadot - OMEGA_E) * tk - OMEGA_E * nav.toe

    cosO = np.cos(Omega)
    sinO = np.sin(Omega)
    cosi = np.cos(i)
    sini = np.sin(i)

    x = x_prime * cosO - y_prime * cosi * sinO
    y = x_prime * sinO + y_prime * cosi * cosO
    z = y_prime * sini

    # Compute satellite clock correction at transmission time
    _, toc_sow = datetime_to_gps_week_seconds(nav.toc)
    dt = _wrap_time_diff(sow - toc_sow)
    dtsv = (
        nav.af0 + nav.af1 * dt + nav.af2 * dt**2 + F_REL * nav.e * nav.sqrtA * np.sin(E)
    )

    return np.array([x, y, z]), dtsv


def compute_satellite_state(
    nav: GPSEphemeris,
    recv_dt: datetime,
    pseudorange_m: float,
) -> tuple[np.ndarray, float]:
    """
    Compute satellite position and clock bias at observation time using broadcast ephemeris.

    Args:
        nav: Broadcast ephemeris
        recv_dt: Reception/observation time (UTC datetime)
        pseudorange_m: Measured pseudorange (meters)

    Returns:
        Tuple of (position [x,y,z] in ECEF meters, clock bias in seconds)
    """
    _, sow = datetime_to_gps_week_seconds(
        recv_dt - timedelta(seconds=pseudorange_m / CLIGHT)
    )
    sat_pos, dtsv = broadcast_ecef_and_clock(nav, sow)
    return sat_pos, dtsv
