import matplotlib.pyplot as plt
import numpy as np


def plot_gps_ca_theoretical_psd(bw_mhz: float = 8.0) -> None:
    fc = 1.023  # C/A code chip rate (1.023 MHz)
    freq = np.linspace(-bw_mhz, bw_mhz, 1000)

    psd_linear = np.sinc(freq / fc) ** 2
    psd_db = 10 * np.log10(psd_linear + 1e-12)

    # Autocorrelation of a rectangular chip: triangle over [-Tc, +Tc]
    Tc = 1.0 / fc  # chip period in µs
    tau = np.linspace(-2 * Tc, 2 * Tc, 1000)
    acf = np.maximum(1.0 - np.abs(tau) / Tc, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("GPS C/A Code — Theoretical PSD and Autocorrelation")

    # --- PSD (top) ---
    ax_psd = axes[0]
    ax_psd.plot(freq, psd_db, color="red", lw=2, label="Theoretical PSD")
    ax_psd.axvline(x=fc, color="black", linestyle="--", alpha=0.6, label=r"Nulls ($\pm$1.023 MHz)")
    ax_psd.axvline(x=-fc, color="black", linestyle="--", alpha=0.6)
    ax_psd.set_title("Power Spectral Density", fontsize=12)
    ax_psd.set_xlabel("Frequency Offset from Carrier (MHz)")
    ax_psd.set_ylabel("Relative PSD (dB)")
    ax_psd.set_xlim(-bw_mhz, bw_mhz)
    ax_psd.set_ylim(-40, 2)
    ax_psd.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_psd.legend()

    # --- Autocorrelation (bottom) ---
    ax_acf = axes[1]
    ax_acf.plot(tau, acf, color="blue", lw=2, label="Autocorrelation (triangle)")
    ax_acf.axvline(x=Tc, color="black", linestyle="--", alpha=0.6, label=r"$\pm T_c$ = ±0.977 µs")
    ax_acf.axvline(x=-Tc, color="black", linestyle="--", alpha=0.6)
    ax_acf.set_xlabel(r"Time Delay $\tau$ (µs)")
    ax_acf.set_ylabel("Normalized Autocorrelation")
    ax_acf.set_title("Autocorrelation Function", fontsize=12)
    ax_acf.set_xlim(-2 * Tc, 2 * Tc)
    ax_acf.set_ylim(-0.7, 1.1)
    ax_acf.axhline(y=0, color="black", linestyle="--", alpha=0.6)
    ax_acf.set_aspect("equal", adjustable="box")
    ax_acf.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_acf.legend()

    plt.tight_layout()
    out_figname = f"gps_ca_theoretical_psd_bw{bw_mhz:.1f}MHz.png"
    plt.savefig(out_figname, dpi=300)
    print(f"Saved figure: {out_figname}")
    plt.show()


if __name__ == "__main__":
    plot_gps_ca_theoretical_psd(bw_mhz=8.0)
