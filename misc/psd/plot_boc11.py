import matplotlib.pyplot as plt
import numpy as np


def plot_gps_boc_theoretical_psd(bw_mhz: float = 8.0):
    # f0 = 1.023                      # fundamental frequency (1.023 MHz)
    fc = 1.023  # code chip rate (BOC(1,1) -> 1.023 MHz)
    fs = 1.023  # subcarrier frequency (BOC(1,1) -> 1.023 MHz)

    # BOC modulation index (number of subcarrier half-cycles per chip)
    # For BOC(1,1), it is 2 * 1.023 / 1.023 = 2
    # N = int(2 * fs / fc)

    # Frequency offset axis (-4 MHz to +4 MHz)
    freq = np.linspace(-bw_mhz, bw_mhz, 2000)
    # Small value to prevent division by zero
    eps = 1e-15
    f_safe = np.where(freq == 0, eps, freq)

    # --- 1. Theoretical PSD of conventional BPSK (C/A code) ---
    psd_bpsk = np.sinc(f_safe / fc) ** 2

    # --- 2. Theoretical PSD of BOC(1,1) ---
    # BOC theoretical formula for even N:
    # G_BOC(f) = fc * ( sin(pi*f/fc) * sin(pi*f/(2*fs)) / (pi*f) )^2 * tan^2(pi*f/(2*fs))
    # *Note: The coefficient fc is excluded to normalize the maximum value to 1
    num = np.sin(np.pi * f_safe / fc) * np.sin(np.pi * f_safe / (2 * fs))
    den = np.pi * f_safe
    term1 = (num / den) ** 2
    term2 = np.tan(np.pi * f_safe / (2 * fs)) ** 2
    psd_boc = term1 * term2

    # --- 3. Convert to dB (equal-power normalization, reference = BPSK peak) ---
    # sinc²(f/fc) integrates to fc; psd_boc formula integrates to 3/(4·fc)
    # Dividing by those constants makes both unit power; BPSK peak = sinc²(0)/fc·fc = 1 → 0 dB
    psd_bpsk_db = 10 * np.log10(psd_bpsk + 1e-12)  # sinc²(0) = 1 → 0 dB
    psd_boc_db = 10 * np.log10(psd_boc * (4 * fc**2 / 3) + 1e-12)

    # --- 4. Autocorrelation functions ---
    Tc = 1.0 / fc  # chip period in µs
    tau = np.linspace(-2 * Tc, 2 * Tc, 2000)
    abs_tau = np.abs(tau)

    # BPSK: triangle over [-Tc, +Tc]
    acf_bpsk = np.maximum(1.0 - abs_tau / Tc, 0.0)

    # BOC(1,1): piecewise linear with secondary minimum at ±Tc/2
    acf_boc = np.where(
        abs_tau <= Tc / 2,
        1.0 - 3.0 * abs_tau / Tc,
        np.where(abs_tau <= Tc, -1.0 + abs_tau / Tc, 0.0),
    )

    # --- 5. Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Theoretical Comparison: BPSK vs BOC(1,1)", fontsize=13)

    # --- PSD ---
    ax_psd = axes[0]
    ax_psd.plot(freq, psd_bpsk_db, color="blue", lw=1.5, linestyle="--", label="Conventional BPSK / C/A")
    ax_psd.plot(freq, psd_boc_db, color="red", lw=2.5, label="BOC(1,1) Theoretical")
    ax_psd.axvline(x=fs, color="black", linestyle=":", alpha=0.7, label="Subcarrier Freq ($\\pm$1.023 MHz)")
    ax_psd.axvline(x=-fs, color="black", linestyle=":", alpha=0.7)
    ax_psd.set_title("Power Spectral Density", fontsize=12)
    ax_psd.set_xlabel("Frequency Offset from Carrier (MHz)", fontsize=11)
    ax_psd.set_ylabel("Relative PSD (dB)", fontsize=11)
    ax_psd.set_xlim(-bw_mhz, bw_mhz)
    ax_psd.set_ylim(-40, 2)
    ax_psd.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_psd.legend(loc="upper right")

    # --- Autocorrelation ---
    ax_acf = axes[1]
    ax_acf.plot(tau, acf_bpsk, color="blue", lw=1.5, linestyle="--", label="BPSK / C/A (triangle)")
    ax_acf.plot(tau, acf_boc, color="red", lw=2.5, label="BOC(1,1)")
    ax_acf.axvline(x=Tc / 2, color="gray", linestyle=":", alpha=0.7, label=r"$\pm T_c/2$ (BOC side lobe)")
    ax_acf.axvline(x=-Tc / 2, color="gray", linestyle=":", alpha=0.7)
    ax_acf.axvline(x=Tc, color="black", linestyle="--", alpha=0.6, label=r"$\pm T_c$ = ±0.977 µs")
    ax_acf.axvline(x=-Tc, color="black", linestyle="--", alpha=0.6)
    ax_acf.axhline(y=0, color="black", linestyle="-", lw=0.8, alpha=0.4)
    ax_acf.set_title("Autocorrelation Function", fontsize=12)
    ax_acf.set_xlabel(r"Time Delay $\tau$ (µs)", fontsize=11)
    ax_acf.set_ylabel("Normalized Autocorrelation", fontsize=11)
    ax_acf.set_xlim(-2 * Tc, 2 * Tc)
    ax_acf.set_ylim(-0.7, 1.1)
    ax_acf.axhline(y=0, color="black", linestyle="--", alpha=0.6)
    ax_acf.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_acf.legend(loc="upper right")
    ax_acf.set_aspect("equal", adjustable="box")

    figamename = f"gps_boc_theoretical_psd_bw{bw_mhz:.1f}MHz.png"
    plt.savefig(figamename, dpi=300)
    print(f"Saved figure: {figamename}")
    plt.show()


if __name__ == "__main__":
    plot_gps_boc_theoretical_psd(bw_mhz=8.0)
