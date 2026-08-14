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


def plot_boc_psd(bw_mhz: float = 20.0) -> None:
    f0 = 1.023  # base chip rate (MHz)
    freq = np.linspace(-bw_mhz, bw_mhz, 4000)
    eps = 1e-15
    f_safe = np.where(np.abs(freq) < eps, eps, freq)

    # BPSK reference (C/A, fc = 1.023 MHz)
    psd_bpsk = np.sinc(f_safe / f0) ** 2

    signals = [
        ("BOC(1,1)", 1, 1, "red", 2.5, "-"),
        ("BOC(10,5)", 10, 5, "green", 2.0, "-"),
        ("BOC(15,10)", 15, 10, "purple", 2.0, "-"),
    ]

    def to_db(psd: np.ndarray) -> np.ndarray:
        return 10 * np.log10(psd / np.max(psd) + 1e-12)

    plt.figure(figsize=(12, 6))
    plt.plot(freq, to_db(psd_bpsk), color="blue", lw=1.5, linestyle="--", label="BPSK / C/A (1.023 Mcps)")

    for label, m, n, color, lw, ls in signals:
        psd = boc_psd(freq, m, n, f0)
        plt.plot(freq, to_db(psd), color=color, lw=lw, linestyle=ls, label=label)
        fs = m * f0
        plt.axvline(x=fs, color=color, linestyle=":", alpha=0.4)
        plt.axvline(x=-fs, color=color, linestyle=":", alpha=0.4)

    plt.title("Theoretical PSD: BPSK vs BOC variants", fontsize=12)
    plt.xlabel("Frequency Offset from Carrier (MHz)", fontsize=11)
    plt.ylabel("Relative PSD (dB, normalized to peak)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.xlim(-bw_mhz, bw_mhz)
    plt.ylim(-40, 2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_boc_psd(bw_mhz=50.0)
