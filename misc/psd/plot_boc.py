import matplotlib.pyplot as plt
import numpy as np


def boc_psd(freq: np.ndarray, m: int, n: int, f0: float = 1.023) -> np.ndarray:
    """BOC(m, n) normalized PSD.

    N = 2m/n (subcarrier half-periods per chip).
    Even N: G ∝ sin²(πf/fc) · tan²(πf/(2fs)) / (πf)²
    Odd  N: G ∝ cos²(πf/fc) · tan²(πf/(2fs)) / (πf)²
    """
    fs = m * f0  # subcarrier frequency (MHz)
    fc = n * f0  # code chip rate (MHz)
    N = (2 * m) // n

    eps = 1e-15
    f_safe = np.where(np.abs(freq) < eps, eps, freq)

    alpha = np.pi * f_safe / fc  # πf/fc
    beta = np.pi * f_safe / (2 * fs)  # πf/(2fs)

    trig_fc = np.sin(alpha) if N % 2 == 0 else np.cos(alpha)

    # sin(beta)*tan(beta) = sin²(beta)/cos(beta); guard against 0/0 at the removable poles
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = (trig_fc * np.sin(beta) / (np.pi * f_safe * np.cos(beta))) ** 2
    psd = np.where(np.isfinite(psd), psd, 0.0)
    return psd


def boc_acf(m: int, n: int, tau_norm: np.ndarray) -> np.ndarray:
    """BOC(m,n) ACF vs normalised delay τ/Tc, computed from the chip waveform."""
    N = (2 * m) // n
    pts = N * 500  # N divides pts exactly
    sub = np.arange(pts) // (pts // N)
    h = np.where(sub % 2 == 0, 1.0, -1.0)
    acf = np.zeros(len(tau_norm))
    for i, tau in enumerate(tau_norm):
        k = int(round(abs(tau) * pts))
        if k >= pts:
            acf[i] = 0.0
        else:
            acf[i] = np.dot(h[k:], h[: pts - k]) / pts
    return acf


def plot_boc_psd(bw_mhz: float = 20.0) -> None:
    f0 = 1.023  # base chip rate (MHz)
    freq = np.linspace(-bw_mhz, bw_mhz, 4000)
    eps = 1e-15
    f_safe = np.where(np.abs(freq) < eps, eps, freq)

    psd_bpsk = np.sinc(f_safe / f0) ** 2

    tau_norm = np.linspace(-2, 2, 2000)  # τ / Tc (dimensionless)
    acf_bpsk = np.maximum(1.0 - np.abs(tau_norm), 0.0)

    signals = [
        ("BOC(1,1)", 1, 1, "red", 2.5, "-"),
        ("BOC(10,5)", 10, 5, "green", 2.0, "-"),
        ("BOC(15,10)", 15, 10, "purple", 2.0, "-"),
    ]

    def to_db(psd: np.ndarray) -> np.ndarray:
        return 10 * np.log10(psd / np.max(psd) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Theoretical Comparison: BPSK vs BOC variants", fontsize=13)

    ax_psd = axes[0]
    ax_psd.plot(freq, to_db(psd_bpsk), color="blue", lw=1.5, linestyle="--", label="BPSK / C/A (1.023 Mcps)")

    ax_acf = axes[1]
    ax_acf.plot(tau_norm, acf_bpsk, color="blue", lw=1.5, linestyle="--", label="BPSK / C/A")

    for label, m, n, color, lw, ls in signals:
        psd = boc_psd(freq, m, n, f0)
        ax_psd.plot(freq, to_db(psd), color=color, lw=lw, linestyle=ls, label=label)
        fs = m * f0
        ax_psd.axvline(x=fs, color=color, linestyle=":", alpha=0.4)
        ax_psd.axvline(x=-fs, color=color, linestyle=":", alpha=0.4)

        acf = boc_acf(m, n, tau_norm)
        ax_acf.plot(tau_norm, acf, color=color, lw=lw, linestyle=ls, label=label)

    ax_psd.set_title("Power Spectral Density", fontsize=12)
    ax_psd.set_xlabel("Frequency Offset from Carrier (MHz)", fontsize=11)
    ax_psd.set_ylabel("Relative PSD (dB, normalized to peak)", fontsize=11)
    ax_psd.set_xlim(-bw_mhz, bw_mhz)
    ax_psd.set_ylim(-40, 2)
    ax_psd.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_psd.legend(loc="upper right")

    ax_acf.axhline(y=0, color="black", linestyle="--", alpha=0.6)
    ax_acf.set_title("Autocorrelation Function", fontsize=12)
    ax_acf.set_xlabel(r"Normalised Delay $\tau\,/\,T_c$", fontsize=11)
    ax_acf.set_ylabel("Normalized Autocorrelation", fontsize=11)
    ax_acf.set_xlim(-2, 2)
    ax_acf.set_ylim(-0.7, 1.1)
    ax_acf.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_acf.legend(loc="upper right")

    plt.tight_layout()
    figname = f"gps_boc_variants_bw{bw_mhz:.0f}MHz.png"
    plt.savefig(figname, dpi=300)
    print(f"Saved figure: {figname}")
    plt.show()


if __name__ == "__main__":
    plot_boc_psd(bw_mhz=50.0)
