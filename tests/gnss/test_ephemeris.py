import json
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import pytest

from app.gnss.ephemeris import (
    GPSEphemeris,
    compute_satellite_state,
    read_rinex_nav,
)

# Check if nav file exists for conditional test skipping
NAV_FILE = Path(__file__).parent / "_tmp_test.nav"
HAS_NAV_FILE = NAV_FILE.exists()

satellite_position_ref = {
    10: {
        "pos": [-5372338.378, 19589602.366, 17185523.845],
        "clk": -576787.986e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 929582),
    },
    12: {
        "pos": [-22523734.631, 6741272.917, 12043745.540],
        "clk": -605651.140e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 931034),
    },
    23: {
        "pos": [-16427074.383, 20026047.569, 5266026.656],
        "clk": 590301.349e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 930515),
    },
    24: {
        "pos": [-16401697.705, -1885765.376, 20341103.798],
        "clk": -189395.996e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 928244),
    },
    25: {
        "pos": [-20044072.138, 17090413.641, 1661220.319],
        "clk": 454156.066e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 929224),
    },
    32: {
        "pos": [4010924.122, 16360271.711, 20843456.254],
        "clk": -193266.388e-9,
        "datetime": datetime(2025, 12, 24, 22, 59, 59, 923578),
    },
}
approx_pos = [-3914831.0330, 3449325.4664, 3656431.3786]  # 3075
# 4 2025/12/24 22:59:59.929582 sat=10 rs= -5372338.378  19589602.366  17185523.845 dts= -576787.986 var=  5.760
# 4 2025/12/24 22:59:59.931034 sat=12 rs=-22523734.631   6741272.917  12043745.540 dts= -605651.140 var=  5.760
# 4 2025/12/24 22:59:59.930515 sat=23 rs=-16427074.383  20026047.569   5266026.656 dts=  590301.349 var=  5.760
# 4 2025/12/24 22:59:59.928244 sat=24 rs=-16401697.705  -1885765.376  20341103.798 dts= -189395.996 var=  5.760
# 4 2025/12/24 22:59:59.929224 sat=25 rs=-20044072.138  17090413.641   1661220.319 dts=  454156.066 var=  5.760
# 4 2025/12/24 22:59:59.923578 sat=32 rs=  4010924.122  16360271.711  20843456.254 dts= -193266.388 var=  5.760

# 4  kepler: sat=10 e= 0.01095 n= 3 del= 2.220e-15
# 4 eph2pos : time=2025/12/24 22:59:59.930 sat=10
# 4 kepler: sat=10 e= 0.01095 n= 3 del= 1.998e-15
# 4 eph2pos : time=2025/12/24 22:59:59.931 sat=12
# 4 kepler: sat=12 e= 0.00904 n= 3 del= 1.110e-16
# 4 eph2pos : time=2025/12/24 22:59:59.932 sat=12
# 4 kepler: sat=12 e= 0.00904 n= 3 del= 1.110e-16
# 4 seleph  : time=2025/12/24 23:00:00.000 sat=23 iode=-1
# 4 eph2pos : time=2025/12/24 22:59:59.930 sat=23
# 4 kepler: sat=23 e= 0.00598 n= 3 del= 0.000e+00
# 4 eph2pos : time=2025/12/24 22:59:59.931 sat=23
# 4 kepler: sat=23 e= 0.00598 n= 3 del= 0.000e+00
# 4 seleph  : time=2025/12/24 23:00:00.000 sat=24 iode=-1
# 4 eph2pos : time=2025/12/24 22:59:59.928 sat=24
# 4 kepler: sat=24 e= 0.01779 n= 3 del=-3.220e-15
# 4 eph2pos : time=2025/12/24 22:59:59.929 sat=24
# 4 kepler: sat=24 e= 0.01779 n= 3 del=-3.220e-15
# 4 seleph  : time=2025/12/24 23:00:00.000 sat=25 iode=-1
# 4 eph2pos : time=2025/12/24 22:59:59.929 sat=25
# 4 kepler: sat=25 e= 0.01288 n= 3 del= 2.887e-15
# 4 eph2pos : time=2025/12/24 22:59:59.930 sat=25
# 4 kepler: sat=25 e= 0.01288 n= 3 del= 2.887e-15
# 4 seleph  : time=2025/12/24 23:00:00.000 sat=32 iode=-1
# 4 eph2pos : time=2025/12/24 22:59:59.923 sat=32
# 4 kepler: sat=32 e= 0.00907 n= 3 del= 4.441e-16
# 4 eph2pos : time=2025/12/24 22:59:59.924 sat=32
# 4 kepler: sat=32 e= 0.00907 n= 3 del= 0.000e+00

eph_ref = {
    10: {
        "prn": 10,
        "toc": "2025-12-24T22:00:00",
        "toe": 338400.0,
        "week": 2398,
        "af0": -0.0005768002010882,
        "af1": -3.637978807092e-12,
        "af2": 0.0,
        "sqrtA": 5153.649131775,
        "e": 0.01094740151893,
        "i0": 0.9919651571362,
        "Omega0": -0.9109940794972,
        "omega": -2.277467574237,
        "M0": -2.248935332041,
        "delta_n": 3.848017428229e-09,
        "Omegadot": -7.833897741857e-09,
        "idot": 1.203621564241e-10,
        "cuc": 3.837049007416e-07,
        "cus": 2.92994081974e-06,
        "crc": 339.53125,
        "crs": 7.8125,
        "cic": 9.126961231232e-08,
        "cis": -8.009374141693e-08,
        "tgd": 0.0,
        "iodc": 333990.0,
        "iode": 12.0,
    },
    12: {
        "prn": 12,
        "toc": "2025-12-24T22:00:00",
        "toe": 338400.0,
        "week": 0,
        "af0": -0.0006056660786271,
        "af1": -5.684341886081e-13,
        "af2": 0.0,
        "sqrtA": 5153.571399689,
        "e": 0.009041805169545,
        "i0": 0.9591414681606,
        "Omega0": 2.289589256388,
        "omega": 1.529666285518,
        "M0": -1.449618117545,
        "delta_n": 5.014137430379e-09,
        "Omegadot": -8.20355599675e-09,
        "idot": 1.407201472733e-10,
        "cuc": 1.439824700356e-06,
        "cus": 6.187707185745e-06,
        "crc": 262.875,
        "crs": 22.25,
        "cic": -1.676380634308e-08,
        "cis": 2.086162567139e-07,
        "tgd": 0.0,
        "iodc": 332670.0,
        "iode": 62.0,
    },
    23: {
        "prn": 23,
        "toc": "2025-12-24T22:00:00",
        "toe": 338400.0,
        "week": 0,
        "af0": 0.000590275041759,
        "af1": 5.115907697473e-12,
        "af2": 0.0,
        "sqrtA": 5153.694128036,
        "e": 0.005978953209706,
        "i0": 0.9869626721912,
        "Omega0": -0.9427659947421,
        "omega": -2.760770741914,
        "M0": -1.140068237584,
        "delta_n": 4.00302388487e-09,
        "Omegadot": -8.029620180196e-09,
        "idot": 4.321608583773e-11,
        "cuc": 6.482005119324e-07,
        "cus": 3.093853592873e-06,
        "crc": 336.1875,
        "crs": 12.6875,
        "cic": 1.322478055954e-07,
        "cis": -9.872019290924e-08,
        "tgd": 0.0,
        "iodc": 331200.0,
        "iode": 208.0,
    },
    24: {
        "prn": 24,
        "toc": "2025-12-24T22:00:00",
        "toe": 338400.0,
        "week": 0,
        "af0": -0.000189421698451,
        "af1": 1.443822839065e-11,
        "af2": 0.0,
        "sqrtA": 5153.656259537,
        "e": 0.01778785674833,
        "i0": 0.9345611294566,
        "Omega0": 1.064399715721,
        "omega": 1.125892716465,
        "M0": 0.1610456132476,
        "delta_n": 5.098426655649e-09,
        "Omegadot": -8.461066723105e-09,
        "idot": 3.003696544589e-10,
        "cuc": -1.687556505203e-06,
        "cus": 1.003034412861e-05,
        "crc": 178.28125,
        "crs": -31.75,
        "cic": -2.197921276093e-07,
        "cis": 2.253800630569e-07,
        "tgd": 0.0,
        "iodc": 331200.0,
        "iode": 97.0,
    },
    25: {
        "prn": 25,
        "toc": "2025-12-24T22:00:00",
        "toe": 338400.0,
        "week": 0,
        "af0": 0.0004541384987533,
        "af1": -2.273736754432e-12,
        "af2": 0.0,
        "sqrtA": 5153.747577667,
        "e": 0.01287716021761,
        "i0": 0.9473832860423,
        "Omega0": 2.196519392729,
        "omega": 1.145977409896,
        "M0": -1.57097136348,
        "delta_n": 5.205573975908e-09,
        "Omegadot": -8.246414924853e-09,
        "idot": 2.292952653539e-10,
        "cuc": 1.413747668266e-06,
        "cus": 5.709007382393e-06,
        "crc": 260.125,
        "crs": 23.09375,
        "cic": 2.384185791016e-07,
        "cis": 1.229345798492e-07,
        "tgd": 0.0,
        "iodc": 337020.0,
        "iode": 74.0,
    },
    32: {
        "prn": 32,
        "toc": "2025-12-25T00:00:00",
        "toe": 345600.0,
        "week": 0,
        "af0": -0.0001932098530233,
        "af1": 1.648459146963e-11,
        "af2": 0.0,
        "sqrtA": 5153.706890106,
        "e": 0.009072169312276,
        "i0": 0.9674849068983,
        "Omega0": 0.11575258988,
        "omega": -2.040018869923,
        "M0": -2.479544906038,
        "delta_n": 4.161244761119e-09,
        "Omegadot": -7.808539542729e-09,
        "idot": -7.428880871279e-11,
        "cuc": -5.461275577545e-06,
        "cus": 6.474554538727e-06,
        "crc": 258.0625,
        "crs": -106.875,
        "cic": 5.774199962616e-08,
        "cis": -1.136213541031e-07,
        "tgd": 0.0,
        "iodc": 338400.0,
        "iode": 24.0,
    },
}


def test_compute_satellite_state():
    obs_time = datetime(2025, 12, 24, 23, 00, 00, 000000)
    # pseurorange
    obs_values = {
        10: 21283814.641,
        12: 20857031.672,
        23: 20654140.320,
        24: 21568703.836,
        25: 21082067.539,
        32: 22968748.305,
    }
    for sat_id, pseudorange in obs_values.items():
        eph_dict = eph_ref[sat_id]
        eph = GPSEphemeris.from_dict(eph_dict)
        pos, clk = compute_satellite_state(
            nav=eph, recv_dt=obs_time, pseudorange_m=pseudorange
        )
        print(f"Satellite {sat_id}: Position = {pos}, Clock = {clk}")
        ref = satellite_position_ref[sat_id]
        ref_pos = ref["pos"]
        ref_clk = ref["clk"]
        print(f"Reference Position = {ref_pos}, Reference Clock = {ref_clk}")
        pos_diff = np.sqrt(sum((pos[i] - ref_pos[i]) ** 2 for i in range(3)))
        clk_diff = abs(clk - ref_clk)
        print(f"diff pos={pos_diff} clk={clk_diff}")
        # Position threshold: 300m (realistic for broadcast ephemeris differences)
        # Clock threshold: 1 microsecond
        assert clk_diff < 1e-6, (
            f"Satellite {sat_id}: clock difference {clk_diff:.3e} s exceeds threshold"
        )
        assert pos_diff < 2.0, (
            f"Satellite {sat_id}: position difference {pos_diff:.3f} m exceeds threshold pos={pos} pos_ref={ref_pos}"
        )


@pytest.fixture
def sample_ephemeris():
    """Create a sample GPSEphemeris object for testing"""
    eph = GPSEphemeris()
    eph.prn = 10
    eph.toc = datetime(2025, 12, 24, 0, 0, 0)
    eph.af0 = -1.234567e-4
    eph.af1 = 5.678901e-12
    eph.af2 = 0.0
    eph.iode = 42
    eph.crs = 5.0
    eph.delta_n = 1.23e-9
    eph.M0 = 2.345678
    eph.cuc = 1.23e-6
    eph.e = 0.01234567
    eph.cus = 2.34e-6
    eph.sqrtA = 5153.5
    eph.toe = 345600.0
    eph.cic = 3.45e-8
    eph.Omega0 = 1.234567
    eph.cis = 4.56e-8
    eph.i0 = 0.987654
    eph.crc = 200.0
    eph.omega = 0.567890
    eph.Omegadot = -7.89e-9
    eph.idot = 1.23e-10
    eph.week = 2345
    eph.tgd = -5.67e-9
    eph.iodc = 42
    return eph


class TestGPSEphemeris:
    """Test suite for the GPSEphemeris class"""

    def test_initialization(self, sample_ephemeris):
        """Test GPSEphemeris object initialization"""
        assert sample_ephemeris.prn == 10
        assert sample_ephemeris.af0 == -1.234567e-4
        assert sample_ephemeris.e == 0.01234567
        assert sample_ephemeris.week == 2345

    def test_str_representation(self, sample_ephemeris):
        """Test string representation of GPSEphemeris"""
        str_repr = str(sample_ephemeris)
        assert "GPSEphemeris" in str_repr
        assert "PRN=10" in str_repr
        assert "TOE=" in str_repr

    def test_repr(self, sample_ephemeris):
        """Test repr representation of GPSEphemeris"""
        repr_str = repr(sample_ephemeris)
        assert "GPSEphemeris" in repr_str
        assert "PRN=10" in repr_str

    def test_to_dict(self, sample_ephemeris):
        """Test conversion to dictionary"""
        data = sample_ephemeris.to_dict()

        assert isinstance(data, dict)
        assert data["prn"] == 10
        assert data["af0"] == -1.234567e-4
        assert data["e"] == 0.01234567
        assert data["week"] == 2345
        assert data["toc"] == "2025-12-24T00:00:00"
        assert "sqrtA" in data
        assert "Omega0" in data

    def test_to_json(self, sample_ephemeris):
        """Test conversion to JSON string"""
        json_str = sample_ephemeris.to_json()

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["prn"] == 10
        assert data["toc"] == "2025-12-24T00:00:00"

    def test_from_dict(self, sample_ephemeris):
        """Test creating GPSEphemeris from dictionary"""
        data = sample_ephemeris.to_dict()
        restored = GPSEphemeris.from_dict(data)

        assert restored.prn == sample_ephemeris.prn
        assert restored.af0 == sample_ephemeris.af0
        assert restored.e == sample_ephemeris.e
        assert restored.toc == sample_ephemeris.toc
        assert restored.week == sample_ephemeris.week

    def test_from_json(self, sample_ephemeris):
        """Test creating GPSEphemeris from JSON string"""
        json_str = sample_ephemeris.to_json()
        restored = GPSEphemeris.from_json(json_str)

        assert restored.prn == sample_ephemeris.prn
        assert restored.af0 == sample_ephemeris.af0
        assert restored.e == sample_ephemeris.e
        assert restored.toc == sample_ephemeris.toc

    def test_roundtrip_dict(self, sample_ephemeris):
        """Test round-trip conversion through dictionary"""
        data = sample_ephemeris.to_dict()
        restored = GPSEphemeris.from_dict(data)
        data2 = restored.to_dict()

        assert data == data2

    def test_roundtrip_json(self, sample_ephemeris):
        """Test round-trip conversion through JSON"""
        json_str = sample_ephemeris.to_json()
        restored = GPSEphemeris.from_json(json_str)
        json_str2 = restored.to_json()

        assert json.loads(json_str) == json.loads(json_str2)


class TestReadRinexNav:
    """Test suite for read_rinex_nav function"""

    @pytest.mark.skipif(not HAS_NAV_FILE, reason=f"NAV file not found: {NAV_FILE}")
    def test_read_existing_file(self):
        """Test reading a real RINEX nav file"""
        nav_file = str(NAV_FILE)
        ephemeris_dict, ion_params = read_rinex_nav(nav_file)

        assert isinstance(ephemeris_dict, dict)
        assert len(ephemeris_dict) > 0

        # Check that all values are lists of GPSEphemeris objects
        for sat_id, eph_list in ephemeris_dict.items():
            assert isinstance(sat_id, str)
            assert isinstance(eph_list, list)
            assert len(eph_list) > 0
            for eph in eph_list:
                assert isinstance(eph, GPSEphemeris)

    @pytest.mark.skipif(not HAS_NAV_FILE, reason=f"NAV file not found: {NAV_FILE}")
    def test_ephemeris_attributes(self):
        """Test that read ephemeris has all required attributes"""
        nav_file = str(NAV_FILE)
        ephemeris_dict, _ion_params = read_rinex_nav(nav_file)

        if ephemeris_dict:
            # Get first ephemeris list and first item
            first_eph_list = next(iter(ephemeris_dict.values()))
            first_eph = first_eph_list[0]

            # Check all required attributes exist
            assert hasattr(first_eph, "prn")
            assert hasattr(first_eph, "toc")
            assert hasattr(first_eph, "af0")
            assert hasattr(first_eph, "e")
            assert hasattr(first_eph, "sqrtA")
            assert hasattr(first_eph, "toe")
            assert hasattr(first_eph, "week")

    def test_mixed_gps_qzss_preserves_system_prefix(self):
        """Mixed NAV should keep satellite IDs as Gxx/Jxx keys."""

        def _line(values: list[float], prefix: str = "    ") -> str:
            return prefix + "".join(f"{v:19.12E}".replace("E", "D") for v in values)

        def _record(system_code: str, prn: int) -> list[str]:
            first = (
                f"{system_code}{prn:02d} 2026 06 13 10 00 00"
                + f"{0.0:19.12E}".replace("E", "D")
                + f"{0.0:19.12E}".replace("E", "D")
                + f"{0.0:19.12E}".replace("E", "D")
            )
            return [
                first,
                _line([0.0, 0.0, 0.0, 0.0]),
                _line([0.0, 0.01, 0.0, 5153.6]),
                _line([0.0, 0.0, 0.0, 0.0]),
                _line([0.0, 0.0, 0.0, 0.0]),
                _line([0.0, 0.0, 0.0, 0.0]),
                _line([0.0, 2397.0, 0.0, 0.0]),
                _line([0.0, 0.0, 0.0, 0.0]),
            ]

        nav_text = (
            "     3.05           N: GNSS NAV DATA    M                   RINEX VERSION / TYPE\n"
            "                                                            END OF HEADER\n"
            + "\n".join(_record("G", 10))
            + "\n"
            + "\n".join(_record("J", 1))
            + "\n"
        )

        with tempfile.NamedTemporaryFile("w", suffix=".nav", delete=False) as tf:
            tf.write(nav_text)
            nav_path = tf.name

        try:
            ephemeris_dict, _ = read_rinex_nav(nav_path)
        finally:
            Path(nav_path).unlink(missing_ok=True)

        assert "G10" in ephemeris_dict
        assert "J01" in ephemeris_dict
        assert len(ephemeris_dict["G10"]) == 1
        assert len(ephemeris_dict["J01"]) == 1
