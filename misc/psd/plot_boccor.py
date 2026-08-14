import matplotlib.pyplot as plt
import numpy as np


def plot_correlation_functions():
    # chip offset axis (-1.5 chips to +1.5 chips)
    tau = np.linspace(-1.5, 1.5, 1000)

    # 1. BPSK (L1 C/A) autocorrelation function (smooth triangle shape)
    r_bpsk = np.where(np.abs(tau) <= 1.0, 1.0 - np.abs(tau), 0.0)

    # 2. BOC(1,1) autocorrelation function (main peak and side dips)
    # Theoretical formula: |tau| <= 0.5 -> 1 - 3*|tau|, 0.5 < |tau| <= 1.0 -> 2*|tau| - 1
    r_boc = np.zeros_like(tau)
    abs_tau = np.abs(tau)

    mask1 = abs_tau <= 0.5
    r_boc[mask1] = 1.0 - 3.0 * abs_tau[mask1]

    mask2 = (abs_tau > 0.5) & (abs_tau <= 1.0)
    r_boc[mask2] = 2.0 * abs_tau[mask2] - 1.0

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(tau, r_bpsk, color="blue", lw=2, linestyle="--", label="BPSK (L1 C/A) - Smooth Peak")
    plt.plot(tau, r_boc, color="red", lw=2.5, label="BOC(1,1) - Sharp Multi-Peak")

    # Auxiliary lines
    plt.axhline(0, color="gray", lw=0.8, ls=":")
    plt.axvline(0, color="gray", lw=0.8, ls=":")

    # Highlight side peaks with false lock risk
    plt.plot([-0.5, 0.5], [-0.5, -0.5], "ko", label="Side Peaks (False Lock Risk)")

    plt.title("Correlation Function Comparison: BPSK vs BOC(1,1)")
    plt.xlabel("Code Phase Offset (Chips)")
    plt.ylabel("Correlation Value")
    plt.grid(True, alpha=0.3)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-0.7, 1.2)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_correlation_functions()
