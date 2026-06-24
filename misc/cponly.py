"""
Relative positioning using carrier phase measurements only.
"""

from app.gnss.constants import CLIGHT, wlen_L1, wlen_L2, wlen_L5
from dataclasses import dataclass
from pathlib import Path
import datetime as dt

import numpy as np

from app.gnss.database import GnssDatabase, Epoch

freq_L1 = CLIGHT / wlen_L1
freq_L2 = CLIGHT / wlen_L2
freq_L5 = CLIGHT / wlen_L5

@dataclass
class RangeObservation3D:
    """
    Data class to hold range observations for 3D positioning.
    """
    satellite_name: str
    receiver_name: str
    wavelength: float  # Wavelength of the signal in meters
    code_range: float  # Code range measurement in meters
    phase_range: float  # Phase range measurement in cycles
    receiver_position: np.ndarray  # Receiver position (3D)
    satellite_position: np.ndarray  # Satellite positions (3D)

    def __post_init__(self):
        # Convert phase range from cycles to meters
        self.phase_range_meters = self.phase_range * self.wavelength

    def geometric_range(self) -> float:
        """
        Calculate the geometric range between the satellite and receiver positions.

        Returns:
        float: Geometric range in meters.
        """
        return np.linalg.norm(self.satellite_position - self.receiver_position)


def adrange(spos: np.ndarray, rpose: np.ndarray, phase_bias: float, wavelength: float) -> np.ndarray:
    """
    Calculate the adjusted range between satellite and receiver positions.

    Parameters:
    spos (np.ndarray): Satellite position (3D).
    rpose (np.ndarray): Receiver position (3D).
    phase_bias (float): Phase bias in cycles.
    wavelength (float): Wavelength of the signal in meters.

    Returns:
    np.ndarray: Adjusted range in meters.
    """
    # Calculate geometric distance
    geometric_distance = np.linalg.norm(spos - rpose)

    # Calculate adjusted range
    adjusted_range = geometric_distance + phase_bias * wavelength

    return adjusted_range


bands = ["L1"]  # signal bands for band-stacking
wlens = {
    "L1": wlen_L1,
    "L2": wlen_L2,
    "L5": wlen_L5,
    "WL": (wlen_L1 * wlen_L2) / (wlen_L2 - wlen_L1),
}

# rec 1: 990840
# rec 2: 02P115
# epoch 1: 2026-06-13 10:14:30
# epoch 2: 2026-06-13 10:32:00
#satellite_blocks = (["G10", "G15", "G20", "G24", "J03", "J07"])
satellite_blocks = (["G10", "G15", "G20", "G24"], ["J03", "J07"])
satellite_order = [sat for block in satellite_blocks for sat in block]

_DB_DIR = Path(__file__).parent
_DB_REC1 = _DB_DIR / "990840.db"
_DB_REC2 = _DB_DIR / "02P115.db"
_EPOCH1 = dt.datetime(2026, 6, 13, 10, 14, 30)
_EPOCH2 = dt.datetime(2026, 6, 13, 10, 32, 0)


def _load_satpos_and_obs(db_path: Path, epoch_dt: dt.datetime, sat_ids: list) -> tuple:
    """Load satellite positions and observations from a GnssDatabase for a given epoch.

    Returns:
        tuple: (satpos, obs)
            satpos: dict[str, np.ndarray]
                Satellite ECEF positions keyed by satellite ID (e.g. "G10").
                Each value is a shape-(3,) array [x, y, z] in meters.
            obs: dict[str, dict[str, dict[str, float]]]
                Observations keyed by satellite ID, then band (e.g. "L1", "L2"),
                each containing {"code_range": float [m], "phase_range": float [cycles]}.
    """
    db = GnssDatabase(db_path)
    satpos = {}
    obs = {}
    session = db.Session()
    try:
        epoch = session.query(Epoch).filter_by(datetime=epoch_dt).one()
        for sat in epoch.satellites:
            if sat.satellite_id not in sat_ids:
                continue
            sid = sat.satellite_id
            if sat.position is not None:
                satpos[sid] = np.array([sat.position.x, sat.position.y, sat.position.z])
            obs[sid] = {
                sig.band: {"code_range": sig.pseudorange, "phase_range": sig.carrier_phase}
                for sig in sat.signals
            }
    finally:
        session.close()
    return satpos, obs


satpos_epoch1_rec1, obs_epoch1_rec1 = _load_satpos_and_obs(_DB_REC1, _EPOCH1, satellite_order)
satpos_epoch2_rec1, obs_epoch2_rec1 = _load_satpos_and_obs(_DB_REC1, _EPOCH2, satellite_order)
satpos_epoch1_rec2, obs_epoch1_rec2 = _load_satpos_and_obs(_DB_REC2, _EPOCH1, satellite_order)
satpos_epoch2_rec2, obs_epoch2_rec2 = _load_satpos_and_obs(_DB_REC2, _EPOCH2, satellite_order)

rec1_pos = np.array([-3914935.45305263, 3484252.78025698, 3622991.44248281])  # 990840 ECEF from SPP (epoch1)
rec2_pos = np.array([-3913071.40599301, 3483060.03925556, 3626133.3403429])   # P115   ECEF from SPP (epoch1)

rec1_pos_prodct = np.array([-3.9149318352E+06,  3.4842523204E+06,  3.6229899127E+06])
rec2_pos_prodct = np.array([-3.9130665486E+06,  3.4830578480E+06,  3.6261311531E+06])
rec2_pos = rec2_pos_prodct


def add_widelane_to_epoch(obs_epoch: dict) -> dict:
    """
    Add wide-lane phase range to the observations for a given epoch.

    Parameters:
    obs_epoch (dict): Observations for the epoch.
    wlen_L1 (float): Wavelength of L1 signal in meters.
    wlen_L2 (float): Wavelength of L2 signal in meters.

    Returns:
    dict: Updated observations with wide-lane phase range added.
    """
    for sat, obs in obs_epoch.items():
        if "L1" in obs and "L2" in obs:
            # Calculate wide-lane phase in cycles: phi_WL = phi_L1 - phi_L2
            obs["WL"] = {"phase_range": obs["L1"]["phase_range"] - obs["L2"]["phase_range"],
                         "code_range": (freq_L1 * obs["L1"]["code_range"] - freq_L2 * obs["L2"]["code_range"]) / (freq_L1 - freq_L2)}
    return obs_epoch

def doble_differencing_matrix(num_sats: int) -> np.ndarray:
    """
    Create a double differencing matrix for num_sats satellites.
    First row: SD between sat1 and sat2, second row: SD between sat1 and sat3, etc.

    Parameters:
    num_sats (int): Number of satellites.

    Returns:
    np.ndarray: Double differencing matrix of shape (num_sats-1, num_sats).
    """
    D = np.zeros((num_sats - 1, num_sats + num_sats), dtype=int)
    for i in range(num_sats - 1):
        D[i, 0] = 1
        D[i, i + 1] = -1
        D[i, num_sats + 0] = -1
        D[i, num_sats + i + 1] = 1
    return D


def block_doble_differencing_matrix(sat_blocks: tuple[list[str], ...]) -> tuple[np.ndarray, list[str]]:
    """Create one global DD matrix by stacking block-wise DD rows.

    Each block has its own reference satellite (the first satellite in the block).
    """
    flat_sat_order = [sat for block in sat_blocks for sat in block]
    sat_index = {sat: i for i, sat in enumerate(flat_sat_order)}
    m_total = len(flat_sat_order)
    rows: list[np.ndarray] = []

    for block in sat_blocks:
        if len(block) < 2:
            continue
        print(f"Creating DD rows for block: {block}")
        ref_idx = sat_index[block[0]]
        for sat in block[1:]:
            sat_idx = sat_index[sat]
            dd_row = np.zeros(2 * m_total, dtype=int)
            dd_row[ref_idx] = 1
            dd_row[sat_idx] = -1
            dd_row[m_total + ref_idx] = -1
            dd_row[m_total + sat_idx] = 1
            rows.append(dd_row)

    if not rows:
        return np.zeros((0, 2 * m_total), dtype=int), flat_sat_order
    return np.vstack(rows), flat_sat_order

def test_doble_differencing_matrix():
    D = doble_differencing_matrix(4)
    print("Double Differencing Matrix D:")
    print(D)
    assert np.array_equal(D, np.array([[1, -1, 0, 0, -1, 1, 0, 0],
                                       [1, 0, -1, 0, -1, 0, 1, 0],
                                       [1, 0, 0, -1, -1, 0, 0, 1]]))
    print("Assertion passed: D is as expected.")

if "WL" in bands:
    for epoch in [obs_epoch1_rec1, obs_epoch2_rec1, obs_epoch1_rec2, obs_epoch2_rec2]:
        add_widelane_to_epoch(epoch)

# DD observations and code-phase biases per band
dd_matrix, satellite_order = block_doble_differencing_matrix(satellite_blocks)
num_dd = dd_matrix.shape[0]
num_bands = len(bands)
print(f"dd_matrix shape: {dd_matrix.shape}, num_dd: {num_dd}, num_bands: {num_bands}")

if num_dd == 0:
    raise ValueError("At least one satellite block must contain 2 or more satellites for DD.")

# dd_obss_epoch{1,2}[band] -> np.ndarray(num_dd,)  [cycles]
dd_obss_epoch1: dict[str, np.ndarray] = {}
dd_obss_epoch2: dict[str, np.ndarray] = {}
dd_code_phase_biases_epoch1: dict[str, np.ndarray] = {}
dd_code_phase_biases_epoch2: dict[str, np.ndarray] = {}

for b in bands:
    wl = wlens[b]
    phi1 = [obs_epoch1_rec1[s][b]['phase_range'] for s in satellite_order]
    phi1 += [obs_epoch1_rec2[s][b]['phase_range'] for s in satellite_order]
    phi2 = [obs_epoch2_rec1[s][b]['phase_range'] for s in satellite_order]
    phi2 += [obs_epoch2_rec2[s][b]['phase_range'] for s in satellite_order]
    dd_obss_epoch1[b] = dd_matrix @ np.array(phi1)
    dd_obss_epoch2[b] = dd_matrix @ np.array(phi2)
    cpb1 = [obs_epoch1_rec1[s][b]['code_range'] / wl - obs_epoch1_rec1[s][b]['phase_range'] for s in satellite_order]
    cpb1 += [obs_epoch1_rec2[s][b]['code_range'] / wl - obs_epoch1_rec2[s][b]['phase_range'] for s in satellite_order]
    cpb2 = [obs_epoch2_rec1[s][b]['code_range'] / wl - obs_epoch2_rec1[s][b]['phase_range'] for s in satellite_order]
    cpb2 += [obs_epoch2_rec2[s][b]['code_range'] / wl - obs_epoch2_rec2[s][b]['phase_range'] for s in satellite_order]
    dd_code_phase_biases_epoch1[b] = dd_matrix @ np.array(cpb1)
    dd_code_phase_biases_epoch2[b] = dd_matrix @ np.array(cpb2)
    print(f"[{b}] DD phase obs    epoch1: {dd_obss_epoch1[b]}")
    print(f"[{b}] DD phase obs    epoch2: {dd_obss_epoch2[b]}")
    print(f"[{b}] DD code-phase biases epoch1: {dd_code_phase_biases_epoch1[b]}")
    print(f"[{b}] DD code-phase biases epoch2: {dd_code_phase_biases_epoch2[b]}")
print()

# observation fnction
class ecpoch_ranges:
    def __init__(self, satellite_positions_rec, satellite_positions_base, base_position_ecef, wavelength):
        self.satellite_positions_rec = satellite_positions_rec
        self.satellite_positions_base = satellite_positions_base
        self.base_position_ecef = base_position_ecef
        self.wavelength = wavelength

    def __call__(self, x):
        # x is the unknown parameter vector (e.g., receiver position corrections)
        # For simplicity, we will just return the double differenced geometric ranges for now
        geometric_ranges = [np.linalg.norm(self.satellite_positions_rec[sat] - x) for sat in satellite_order]
        geometric_ranges += [np.linalg.norm(self.satellite_positions_base[sat] - self.base_position_ecef) for sat in satellite_order]
        dd_geometric_ranges = dd_matrix @ np.array(geometric_ranges)
        return dd_geometric_ranges

class epoch_line_of_sight:
    def __init__(self, satellite_positions_rec, satellite_positions_base, base_position_ecef, band: str = "L1"):
        self.satellite_positions_rec = satellite_positions_rec
        self.satellite_positions_base = satellite_positions_base
        self.base_position_ecef = base_position_ecef
        self.band = band

    def __call__(self, x):
        # x is the unknown parameter vector (e.g., receiver position corrections)
        # For simplicity, we will just return the double differenced line-of-sight vectors for now
        los_vectors = [(self.satellite_positions_rec[sat] - x) / np.linalg.norm(self.satellite_positions_rec[sat] - x) for sat in satellite_order]
        los_vectors += [(self.satellite_positions_base[sat] - self.base_position_ecef) / np.linalg.norm(self.satellite_positions_base[sat] - self.base_position_ecef) for sat in satellite_order]
        dd_los_vectors = dd_matrix @ np.array(los_vectors)
        return dd_los_vectors

dd_model_epoch1: dict[str, ecpoch_ranges] = {}
dd_model_epoch2: dict[str, ecpoch_ranges] = {}
for b in bands:
    wl = wlens[b]
    dd_model_epoch1[b] = ecpoch_ranges(satpos_epoch1_rec1, satpos_epoch1_rec2, rec2_pos, wl)
    dd_model_epoch2[b] = ecpoch_ranges(satpos_epoch2_rec1, satpos_epoch2_rec2, rec2_pos, wl)

# dd_phase_biases[band] -> np.ndarray(num_dd,)  [cycles]
dd_phase_biases: dict[str, np.ndarray] = {b: np.zeros(num_dd) for b in bands}

# D_left: (num_dd x m) single-difference operator for the rover-side only
m = len(satellite_order)
D_left = dd_matrix[:, :m]

# --- Carrier phase noise and covariance ---
# sigma_phi[band]: 1-sigma carrier phase noise [cycles] for a single raw observation
sigma_phi: dict[str, float] = {"L1": 0.01, "L2": 0.02}  # ~2mm / ~4mm

# DD covariance block for one (band, epoch):
#   C_raw = σ² I_{2m}  →  C_DD = D @ C_raw @ D.T = σ² * (D @ D.T)
#   D @ D.T has diagonal=4, off-diagonal=2  (= 2*I + 2*ones*ones.T)
#
# Full observation layout (row blocks): [L1_ep1, L1_ep2, L2_ep1, L2_ep2, ...]
# Blocks are independent (different epochs / different bands)
#   → full C is block-diagonal; W = C^{-1} is also block-diagonal.
n_obs = 2 * num_bands * num_dd
W = np.zeros((n_obs, n_obs))  # weight matrix (inverse of full covariance)
C_DD_base = dd_matrix @ dd_matrix.T  # shape (num_dd, num_dd), = 2I + 2*11^T
for ib, b in enumerate(bands):
    s2 = sigma_phi[b] ** 2
    C_block = s2 * C_DD_base           # covariance for one (band, epoch) block
    W_block  = np.linalg.inv(C_block)  # weight block
    for ie in range(2):                # 2 epochs
        row = (2 * ib + ie) * num_dd
        W[row:row + num_dd, row:row + num_dd] = W_block
    print(f"DD covariance block ({b}): σ²·DDᵀ =\n{sigma_phi[b]**2 * C_DD_base}")
    print(f"Weight block ({b}):\n{np.linalg.inv(sigma_phi[b]**2 * C_DD_base)}\n")

# Band-stacking layout per iteration (row blocks, grouped by band then epoch):
#   [epoch1_L1(num_dd), epoch2_L1(num_dd), epoch1_L2(num_dd), epoch2_L2(num_dd), ...]
# Unknowns: [d_pos(3), d_N_L1(num_dd), d_N_L2(num_dd), ...]
# Carrier phase equation (cycles): phi = rho/lambda + N
# Jacobian H: (2*num_bands*num_dd) x (3 + num_bands*num_dd)

for itr in range(10):
    dd_rho_ep1 = dd_model_epoch1[b](rec1_pos)  # DD geometric ranges epoch1 [m]
    dd_rho_ep2 = dd_model_epoch2[b](rec1_pos)  # DD geometric ranges epoch2 [m]

    # --- unit LOS vectors from rec1 to each satellite: (m x 3) ---
    los_ep1 = np.array([(satpos_epoch1_rec1[s] - rec1_pos) / np.linalg.norm(satpos_epoch1_rec1[s] - rec1_pos) for s in satellite_order])
    los_ep2 = np.array([(satpos_epoch2_rec1[s] - rec1_pos) / np.linalg.norm(satpos_epoch2_rec1[s] - rec1_pos) for s in satellite_order])

    # --- build stacked residual and Jacobian ---
    n_unk = 3 + num_bands * num_dd
    dy = np.zeros(n_obs)
    H  = np.zeros((n_obs, n_unk))

    for ib, b in enumerate(bands):
        wl = wlens[b]
        row_ep1 = (2 * ib    ) * num_dd
        row_ep2 = (2 * ib + 1) * num_dd
        col_N   = 3 + ib * num_dd
        dy[row_ep1:row_ep1 + num_dd] = dd_obss_epoch1[b] - (dd_rho_ep1 / wl + dd_phase_biases[b])
        dy[row_ep2:row_ep2 + num_dd] = dd_obss_epoch2[b] - (dd_rho_ep2 / wl + dd_phase_biases[b])
        H[row_ep1:row_ep1 + num_dd, :3] = -D_left @ los_ep1 / wl
        H[row_ep2:row_ep2 + num_dd, :3] = -D_left @ los_ep2 / wl
        H[row_ep1:row_ep1 + num_dd, col_N:col_N + num_dd] = np.eye(num_dd)
        H[row_ep2:row_ep2 + num_dd, col_N:col_N + num_dd] = np.eye(num_dd)

    # --- weighted least squares: dx = (H.T W H)^{-1} H.T W dy ---
    HtW = H.T @ W
    dx, *_ = np.linalg.lstsq(HtW @ H, HtW @ dy, rcond=None)

    # --- update unknowns ---
    rec1_pos = rec1_pos + dx[:3]
    for ib, b in enumerate(bands):
        dd_phase_biases[b] += dx[3 + ib * num_dd:3 + (ib + 1) * num_dd]

    norm_pos  = np.linalg.norm(dx[:3])
    norm_bias = np.linalg.norm(dx[3:])
    print(f"Iteration {itr + 1}: |d_pos|={norm_pos:.6f} m  |d_bias|={norm_bias:.6f} cyc (|dy|^2={np.linalg.norm(dy)**2:e})")
    print("  Updated rec1_pos: ", rec1_pos)
    #if norm_pos < 1e-4 and norm_bias < 1e-4:
    #    print("  Converged.")
    #    break

print("\n=== Result ===")
print(f"{bands=}")
print(f"satellite blocks: {satellite_blocks}")
print(f"rec1_pos      : {rec1_pos}")
print("rec1_pos_product : ", rec1_pos_prodct)
print(f"difference from rec1_pos_product: {rec1_pos - rec1_pos_prodct} ({np.linalg.norm(rec1_pos - rec1_pos_prodct):.3f} m)")
for b in bands:
    print(f"dd_phase_biases[{b}]: {dd_phase_biases[b]}")
