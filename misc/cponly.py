"""
Relative positioning using carrier phase measurements only.
"""

from app.gnss.constants import CLIGHT, wlen_L1, wlen_L2, wlen_L5
from dataclasses import dataclass

import numpy as np

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


def adrange(
    spos: np.ndarray, rpose: np.ndarray, phase_bias: float, wavelength: float
) -> np.ndarray:
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


band = "L1"  # signal band to use for positioning (e.g., "L1", "L2", "L5", "WL")
if band == "L1":
    wlen = wlen_L1
elif band == "L2":
    wlen = wlen_L2
elif band == "L5":
    wlen = wlen_L5
elif band == "WL":
    wlen = (wlen_L1 * wlen_L2) / (wlen_L2 - wlen_L1)  # wide-lane wavelength
else:
    raise ValueError(f"Unsupported band: {band}")

# rec 1: 990840
# rec 2: 02P115
# epoch 1: 2026-06-13 10:00:00.000000
# epoch 2: 2026-06-13 10:30:00.000000
satellite_order = ["G10", "G15", "G20", "G24"]
satpos_epoch1_rec1 = {
    "G10": np.array([4908091.53689808, 13755004.9095124, 22449934.1448391]),
    "G15": np.array([-21225644.4083522, 1129931.47958366, 15399513.7196964]),
    "G20": np.array([-22898548.7116418, -7720813.73938214, 11058541.7029171]),
    "G24": np.array([-14526134.1306901, 10084924.6622295, 19164218.6133564]),
}
satpos_epoch2_rec1 = {
    "G10": np.array([387984.114435919, 15733650.1692594, 21600191.9480676]),
    "G15": np.array([-23923000.0692438, -670424.127093148, 11008515.9506502]),
    "G20": np.array([-24240265.4085439, -9178157.13379645, 5804968.34566844]),
    "G24": np.array([-14902359.9725232, 5249469.09269328, 20766200.5871524]),
}
satpos_epoch1_rec2 = {
    "G10": np.array([4908092.45790845, 13755004.5680647, 22449934.1642992]),
    "G15": np.array([-21225643.7963029, 1129931.91348325, 15399514.5022618]),
    "G20": np.array([-22898548.362314, -7720813.36432408, 11058542.6944613]),
    "G24": np.array([-14526134.0998173, 10084925.5875292, 19164218.1498638]),
}
satpos_epoch2_rec2 = {
    "G10": np.array([387985.93812255, 15733649.2417596, 21600192.6239012]),
    "G15": np.array([-23922999.0821456, -670423.500472476, 11008518.0273937]),
    "G20": np.array([-24240265.0218867, -9178156.67229213, 5804970.71705111]),
    "G24": np.array([-14902359.713356, 5249471.20667135, 20766200.2205344]),
}

obs_epoch1_rec1 = {
    "G10": {
        "L1": {"code_range": 23342205.009, "phase_range": 122664145.216},
        "L2": {"code_range": 23342211.699, "phase_range": 95582486.228},
        "L5": {"code_range": 23342208.44, "phase_range": 91599870.745},
    },
    "G15": {
        "L1": {"code_range": 20921596.908, "phase_range": 109943824.401},
        "L2": {"code_range": 20921602.005, "phase_range": 85670549.439},
    },
    "G20": {
        "L1": {"code_range": 23184686.323, "phase_range": 121836376.731},
        "L2": {"code_range": 23184693.492, "phase_range": 94937481.581},
        "L5": {"code_range": 23184693.771, "phase_range": 90981755.987},
    },
    "G24": {
        "L1": {"code_range": 19903247.377, "phase_range": 104592406.503},
        "L2": {"code_range": 19903253.548, "phase_range": 81500661.712},
        "L5": {"code_range": 19903251.296, "phase_range": 78104811.508},
    },
}
obs_epoch2_rec1 = {
    "G10": {
        "L1": {"code_range": 22241813.380, "phase_range": 116881557.185},
        "L2": {"code_range": 22241819.876, "phase_range": 91076576.209},
        "L5": {"code_range": 22241815.746, "phase_range": 87281707.529},
    },
    "G15": {
        "L1": {"code_range": 21496050.754, "phase_range": 112962591.141},
        "L2": {"code_range": 21496055.079, "phase_range": 88022833.443},
    },
    "G20": {
        "L1": {"code_range": 23881314.038, "phase_range": 125497164.127},
        "L2": {"code_range": 23881321.399, "phase_range": 97790040.164},
        "L5": {"code_range": 23881322.313, "phase_range": 93715457.28},
    },
    "G24": {
        "L1": {"code_range": 20313937.461, "phase_range": 106750597.330},
        "L2": {"code_range": 20313943.898, "phase_range": 83182368.701},
        "L5": {"code_range": 20313941.503, "phase_range": 79716447.323},
    },
}
obs_epoch1_rec2 = {
    "G10": {
        "L1": {"code_range": 23448253.282, "phase_range": 123221435.916},
        "L2": {"code_range": 23448260.677, "phase_range": 96016731.015},
    },
    "G15": {
        "L1": {"code_range": 21030018.605, "phase_range": 110513687.712},
        "L2": {"code_range": 21030023.085, "phase_range": 86114638.507},
    },
    "G20": {
        "L1": {"code_range": 23293408.675, "phase_range": 122407905.610},
        "L2": {"code_range": 23293414.945, "phase_range": 95382898.714},
    },
    "G24": {
        "L1": {"code_range": 20010965.590, "phase_range": 105158627.568},
        "L2": {"code_range": 20010972.252, "phase_range": 81941935.544},
        "L5": {"code_range": 20010966.235, "phase_range": 78527390.82},
    },
}
obs_epoch2_rec2 = {
    "G10": {
        "L1": {"code_range": 22470875.225, "phase_range": 118085291.631},
        "L2": {"code_range": 22470882.184, "phase_range": 92014543.408},
    },
    "G15": {
        "L1": {"code_range": 21727782.167, "phase_range": 114180451.389},
        "L2": {"code_range": 21727786.357, "phase_range": 88971855.147},
    },
    "G20": {
        "L1": {"code_range": 24113286.794, "phase_range": 126716383.236},
        "L2": {"code_range": 24113293.155, "phase_range": 98740150.995},
    },
    "G24": {
        "L1": {"code_range": 20543718.586, "phase_range": 107958262.511},
        "L2": {"code_range": 20543725.561, "phase_range": 84123469.101},
        "L5": {"code_range": 20543719.33, "phase_range": 80618027.094},
    },
}

rec1_pos = np.array(
    [-3914935.45305263, 3484252.78025698, 3622991.44248281]
)  # 990840 ECEF from SPP (epoch1)
rec2_pos = np.array(
    [-3913071.40599301, 3483060.03925556, 3626133.3403429]
)  # P115   ECEF from SPP (epoch1)

rec1_pos_prodct = np.array([-3.9149318352e06, 3.4842523204e06, 3.6229899127e06])
rec2_pos_prodct = np.array([-3.9130665486e06, 3.4830578480e06, 3.6261311531e06])
rec2_pos = rec2_pos_prodct


def add_widelane_to_epoch(obs_epoch: dict, wlen_L1: float, wlen_L2: float) -> dict:
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
            obs["WL"] = {
                "phase_range": obs["L1"]["phase_range"] - obs["L2"]["phase_range"],
                "code_range": (
                    freq_L1 * obs["L1"]["code_range"]
                    - freq_L2 * obs["L2"]["code_range"]
                )
                / (freq_L1 - freq_L2),
            }
    return obs_epoch


def doble_differencing_matrix(m: int) -> np.ndarray:
    """
    Create a double differencing matrix for m satellites.

    Parameters:
    m (int): Number of satellites.

    Returns:
    np.ndarray: Double differencing matrix of shape (m-1, m).
    """
    D = np.zeros((m - 1, m + m), dtype=int)
    for i in range(m - 1):
        D[i, 0] = 1
        D[i, i + 1] = -1
        D[i, m + 0] = -1
        D[i, m + i + 1] = 1
    return D


def test_doble_differencing_matrix():
    D = doble_differencing_matrix(4)
    print("Double Differencing Matrix D:")
    print(D)
    assert np.array_equal(
        D,
        np.array(
            [
                [1, -1, 0, 0, -1, 1, 0, 0],
                [1, 0, -1, 0, -1, 0, 1, 0],
                [1, 0, 0, -1, -1, 0, 0, 1],
            ]
        ),
    )
    print("Assertion passed: D is as expected.")


for epoch in [obs_epoch1_rec1, obs_epoch2_rec1, obs_epoch1_rec2, obs_epoch2_rec2]:
    add_widelane_to_epoch(epoch, wlen_L1, wlen_L2)

# epoch 1 and epoch 2 observations
dd_matrix = doble_differencing_matrix(len(satellite_order))
obss_epoch1: list = []  # list to hold phase ranges for epoch 1
obss_epoch2: list = []  # list to hold phase ranges for epoch 2
for sat in satellite_order:
    print(
        f"Satellite: {sat}, Code Range: {obs_epoch1_rec1[sat][band]['code_range']}, Phase Range (cycles): {obs_epoch1_rec1[sat][band]['phase_range']}"
    )
    obss_epoch1.append(obs_epoch1_rec1[sat][band]["phase_range"])
    obss_epoch2.append(obs_epoch2_rec1[sat][band]["phase_range"])
for sat in satellite_order:
    print(
        f"Satellite: {sat}, Code Range: {obs_epoch1_rec2[sat][band]['code_range']}, Phase Range (cycles): {obs_epoch1_rec2[sat][band]['phase_range']}"
    )
    obss_epoch1.append(obs_epoch1_rec2[sat][band]["phase_range"])
    obss_epoch2.append(obs_epoch2_rec2[sat][band]["phase_range"])
print("Phase ranges for epoch 1:", obss_epoch1)
dd_obss_epoch1: np.ndarray = dd_matrix @ np.array(obss_epoch1)
print("Double differenced observations for epoch 1:", dd_obss_epoch1)
print("Phase ranges for epoch 2:", obss_epoch2)
dd_obss_epoch2: np.ndarray = dd_matrix @ np.array(obss_epoch2)
print("Double differenced observations for epoch 2:", dd_obss_epoch2)
print("")

# code-phase biases for each satellite (in cycles)
code_phase_biases_epoch1 = [
    obs_epoch1_rec1[sat][band]["code_range"] / wlen
    - obs_epoch1_rec1[sat][band]["phase_range"]
    for sat in satellite_order
]
code_phase_biases_epoch1 += [
    obs_epoch1_rec2[sat][band]["code_range"] / wlen
    - obs_epoch1_rec2[sat][band]["phase_range"]
    for sat in satellite_order
]
dd_code_phase_biases_epoch1: np.ndarray = dd_matrix @ np.array(code_phase_biases_epoch1)
# print("Code-phase biases for epoch 1 (in cycles):", code_phase_biases_epoch1)
code_phase_biases_epoch2 = [
    obs_epoch2_rec1[sat][band]["code_range"] / wlen
    - obs_epoch2_rec1[sat][band]["phase_range"]
    for sat in satellite_order
]
code_phase_biases_epoch2 += [
    obs_epoch2_rec2[sat][band]["code_range"] / wlen
    - obs_epoch2_rec2[sat][band]["phase_range"]
    for sat in satellite_order
]
dd_code_phase_biases_epoch2: np.ndarray = dd_matrix @ np.array(code_phase_biases_epoch2)
# print("Code-phase biases for epoch 2 (in cycles):", code_phase_biases_epoch2)
print(
    "Double differenced code-phase biases for epoch 1 (in cycles):",
    dd_code_phase_biases_epoch1,
)
print(
    "Double differenced code-phase biases for epoch 2 (in cycles):",
    dd_code_phase_biases_epoch2,
)
print("")


# observation fnction
class ecpoch_ranges:
    def __init__(
        self,
        satellite_positions_rec,
        satellite_positions_base,
        base_position_ecef,
        wavelength,
    ):
        self.satellite_positions_rec = satellite_positions_rec
        self.satellite_positions_base = satellite_positions_base
        self.base_position_ecef = base_position_ecef
        self.wavelength = wavelength

    def __call__(self, x):
        # x is the unknown parameter vector (e.g., receiver position corrections)
        # For simplicity, we will just return the double differenced geometric ranges for now
        geometric_ranges = [
            np.linalg.norm(self.satellite_positions_rec[sat] - x)
            for sat in satellite_order
        ]
        geometric_ranges += [
            np.linalg.norm(self.satellite_positions_base[sat] - self.base_position_ecef)
            for sat in satellite_order
        ]
        dd_geometric_ranges = dd_matrix @ np.array(geometric_ranges)
        return dd_geometric_ranges


class epoch_line_of_sight:
    def __init__(
        self,
        satellite_positions_rec,
        satellite_positions_base,
        base_position_ecef,
        band: str = "L1",
    ):
        self.satellite_positions_rec = satellite_positions_rec
        self.satellite_positions_base = satellite_positions_base
        self.base_position_ecef = base_position_ecef
        self.band = band

    def __call__(self, x):
        # x is the unknown parameter vector (e.g., receiver position corrections)
        # For simplicity, we will just return the double differenced line-of-sight vectors for now
        los_vectors = [
            (self.satellite_positions_rec[sat] - x)
            / np.linalg.norm(self.satellite_positions_rec[sat] - x)
            for sat in satellite_order
        ]
        los_vectors += [
            (self.satellite_positions_base[sat] - self.base_position_ecef)
            / np.linalg.norm(
                self.satellite_positions_base[sat] - self.base_position_ecef
            )
            for sat in satellite_order
        ]
        dd_los_vectors = dd_matrix @ np.array(los_vectors)
        return dd_los_vectors


dd_model_epoch1 = ecpoch_ranges(satpos_epoch1_rec1, satpos_epoch1_rec2, rec2_pos, wlen)
dd_model_epoch2 = ecpoch_ranges(satpos_epoch2_rec1, satpos_epoch2_rec2, rec2_pos, wlen)
print(
    "Double differenced model predictions for epoch 1:",
    dd_model_epoch1(rec1_pos) + dd_code_phase_biases_epoch1,
)
print(
    "Double differenced model predictions for epoch 2:",
    dd_model_epoch2(rec1_pos) + dd_code_phase_biases_epoch2,
)

dd_phase_biases = (
    -(dd_code_phase_biases_epoch1 + dd_code_phase_biases_epoch2) / 2
)  # average code-phase bias across epochs
dd_phase_biases = np.zeros_like(dd_phase_biases)  # for testing, set biases to zero
num_dd = len(satellite_order) - 1  # number of double-differenced observations per epoch

# D_left: (num_dd x m) single-difference operator for the rover-side only
m = len(satellite_order)
D_left = dd_matrix[:, :m]

# Carrier phase equation (cycles): phi = rho/lambda + N
# Residual: dy = dd_phi_obs - (dd_rho/lambda + dd_N)    [cycles]
# Jacobian: d(rho_j)/d(rec1) = -e_j  (gradient of range w.r.t. receiver pos)
#   => H_pos = -D_left @ los_rec / lambda    (num_dd x 3)

for itr in range(20):
    # --- residual vector (cycles) ---
    dy = np.zeros(num_dd * 2)
    dy[:num_dd] = dd_obss_epoch1 - (dd_model_epoch1(rec1_pos) / wlen + dd_phase_biases)
    dy[num_dd:] = dd_obss_epoch2 - (dd_model_epoch2(rec1_pos) / wlen + dd_phase_biases)

    # --- unit LOS vectors from rec1 to each satellite: (m x 3) ---
    los_ep1 = np.array(
        [
            (satpos_epoch1_rec1[s] - rec1_pos)
            / np.linalg.norm(satpos_epoch1_rec1[s] - rec1_pos)
            for s in satellite_order
        ]
    )
    los_ep2 = np.array(
        [
            (satpos_epoch2_rec1[s] - rec1_pos)
            / np.linalg.norm(satpos_epoch2_rec1[s] - rec1_pos)
            for s in satellite_order
        ]
    )

    # --- Jacobian: (2*num_dd) x (3 + num_dd) ---
    H = np.zeros((num_dd * 2, 3 + num_dd))
    H[0:num_dd, :3] = -D_left @ los_ep1 / wlen  # pos partials epoch 1
    H[num_dd:, :3] = -D_left @ los_ep2 / wlen  # pos partials epoch 2
    H[0:num_dd, 3 : 3 + num_dd] = np.eye(num_dd)  # bias partials
    H[num_dd:, 3 : 3 + num_dd] = np.eye(num_dd)

    # --- solve: dy = H @ [d_pos; d_bias] ---
    dx, *_ = np.linalg.lstsq(H, dy, rcond=None)

    # --- update unknowns ---
    rec1_pos = rec1_pos + dx[:3]
    dd_phase_biases = dd_phase_biases + dx[3:]

    norm_pos = np.linalg.norm(dx[:3])
    norm_bias = np.linalg.norm(dx[3:])
    print(
        f"Iteration {itr + 1}: |d_pos|={norm_pos:.6f} m  |d_bias|={norm_bias:.6f} cyc"
    )

    # if norm_pos < 1e-4 and norm_bias < 1e-4:
    #    print("  Converged.")
    #    break

print("\n=== Result ===")
print(f"{band=} wavelength: {wlen:.3f} m")
print(f"satellite pair order: {satellite_order}")
print(
    f"code-phase biases (cycles): {dd_code_phase_biases_epoch1}, {dd_code_phase_biases_epoch2}"
)
print(f"rec1_pos      : {rec1_pos}")
print("rec1_pos_product : ", rec1_pos_prodct)
print(f"difference from rec1_pos_product: {rec1_pos - rec1_pos_prodct}")
print(f"dd_phase_biases: {dd_phase_biases}")
