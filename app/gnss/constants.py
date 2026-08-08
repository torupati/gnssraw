"""GNSS physical constants and derived wavelengths.

Keep computation separate from plotting, so other modules can import
these values without circular dependencies.
"""

from datetime import datetime, timezone

CLIGHT: float = 299_792_458.0

# L-band carrier frequencies (Hz)
L1_FREQ: float = 1.57542e9
L2_FREQ: float = 1.22760e9
L5_FREQ: float = 1.17645e9
L6_FREQ: float = 1.27875e9
E1_FREQ: float = L1_FREQ  # Galileo E1
E5A_FREQ: float = L5_FREQ  # Galileo E5a
E5B_FREQ: float = 1.20714e9
E6_FREQ: float = L6_FREQ  # Galileo E6
L7_FREQ: float = E5B_FREQ  # GPS L7 same as Galileo E5b
E8_FREQ: float = 1.191795e9  # E5(A+B) blended frequency

# Wavelengths (m)
wlen_L1: float = CLIGHT / L1_FREQ
wlen_L2: float = CLIGHT / L2_FREQ
wlen_L5: float = CLIGHT / L5_FREQ
wlen_L6: float = CLIGHT / L6_FREQ
wlen_E1: float = CLIGHT / E1_FREQ  # Galileo E1
wlen_E5A: float = CLIGHT / E5A_FREQ  # Galileo E5a
wlen_E5B: float = CLIGHT / E5B_FREQ  # Galileo E5b
wlen_E6: float = CLIGHT / E6_FREQ  # Galileo E6
wlen_L7: float = CLIGHT / L7_FREQ
wlen_L8: float = CLIGHT / E8_FREQ  # Galileo E8

### obsolete constants below; kept for reference ###

# Wide- and Narrow-lane effective wavelengths (m)
wl_wlen: float = CLIGHT / (L1_FREQ - L2_FREQ)
nl_wlen: float = CLIGHT / (L1_FREQ + L2_FREQ)

# Iono-free combined effective wavelength (m)
iono_wlen: float = (L1_FREQ**2) / (L1_FREQ**2 - L2_FREQ**2) * wlen_L1 + ((L2_FREQ**2) / (L1_FREQ**2 - L2_FREQ**2)) * wlen_L2

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
