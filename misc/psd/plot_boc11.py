import matplotlib.pyplot as plt
import numpy as np


def plot_gps_boc_theoretical_psd():
    # f0 = 1.023                      # fundamental frequency (1.023 MHz)
    fc = 1.023  # code chip rate (BOC(1,1) -> 1.023 MHz)
    fs = 1.023  # subcarrier frequency (BOC(1,1) -> 1.023 MHz)

    # BOC modulation index (number of subcarrier half-cycles per chip)
    # For BOC(1,1), it is 2 * 1.023 / 1.023 = 2
    # N = int(2 * fs / fc)

    # Frequency offset axis (-4 MHz to +4 MHz)
    freq = np.linspace(-4, 4, 2000)
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

    # --- 3. Convert to decibel (dB) scale (normalize each maximum to 0 dB) ---
    psd_bpsk_db = 10 * np.log10(psd_bpsk / np.max(psd_bpsk) + 1e-12)
    psd_boc_db = 10 * np.log10(psd_boc / np.max(psd_boc) + 1e-12)

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot BPSK (C/A)
    plt.plot(freq, psd_bpsk_db, color="blue", lw=1.5, linestyle="--", label="Conventional BPSK / C/A")

    # Plot BOC(1,1)
    plt.plot(freq, psd_boc_db, color="red", lw=2.5, label="BOC(1,1) Theoretical")

    # Auxiliary lines for the characteristic subcarrier frequency positions (±1.023 MHz)
    plt.axvline(x=fs, color="black", linestyle=":", alpha=0.7, label="Subcarrier Freq ($\pm$1.023 MHz)")
    plt.axvline(x=-fs, color="black", linestyle=":")

    plt.title("Theoretical PSD Comparison: BPSK vs BOC(1,1)", fontsize=12)
    plt.xlabel("Frequency Offset from Carrier (MHz)", fontsize=11)
    plt.ylabel("Relative Power Spectral Density (dB)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.xlim(-4, 4)
    plt.ylim(-40, 2)
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    plot_gps_boc_theoretical_psd()
