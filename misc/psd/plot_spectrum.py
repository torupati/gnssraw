import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftshift


def generate_ca_code_simplified(length):
    """
    Generate a simplified pseudo-random code sequence (C/A code equivalent) of given length.
    In practice, the C/A code is a Gold code, but for the purpose of simulating the spectrum,
    a random ±1 binary pulse sequence suffices to produce the theoretical sinc^2 shape.
    """
    return np.random.choice([-1, 1], size=length)


def plot_gps_ca_spectrum():

    fc = 1.023e6  # C/A code chip rate: 1.023 MHz
    chips_count = 1023  # 1 period chip counter (1 ms 1023 chips)
    oversampling = 20  # oversampling factor (to increase resolution in frequency domain)

    fs = fc * oversampling  # sampling frequency (20.46 MHz)
    total_samples = chips_count * oversampling  # total number of samples
    dt = 1 / fs  # sampling period

    # --- 1. Generate time domain signal ---
    ca_chips = generate_ca_code_simplified(chips_count)
    # Each chip is stretched by the specified number of samples to create a rectangular wave
    signal = np.repeat(ca_chips, oversampling)
    time = np.arange(total_samples) * dt

    # --- 2. Convert to frequency domain (FFT) ---
    # Apply window function (Hamming window) to suppress sidelobe leakage
    window = np.hamming(total_samples)
    fft_signal = fft(signal * window)
    fft_shifted = fftshift(fft_signal)

    # Calculate frequency axis (in MHz)
    frequencies = np.linspace(-fs / 2, fs / 2, total_samples) / 1e6

    # Convert power spectral density (PSD) to decibel (dB) scale
    psd = 20 * np.log10(np.abs(fft_shifted) + 1e-12)
    psd -= np.max(psd)  # Normalize maximum value to 0 dB

    # --- 3. Plotting ---
    plt.figure(figsize=(12, 6))

    # Left: Time domain waveform (zoom in on the first few chips)
    plt.subplot(1, 2, 1)
    plot_samples = oversampling * 10  # Plot the first 10 chips (10 ms)
    plt.step(time[:plot_samples] * 1e6, signal[:plot_samples], where="post", color="blue", lw=2)
    plt.title("Time Domain: C/A Code Signal (First 10 chips)")
    plt.xlabel("Time ($\mu$s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.ylim(-1.5, 1.5)

    # Right: Frequency domain spread spectrum
    plt.subplot(1, 2, 2)
    plt.plot(frequencies, psd, color="red", lw=1.5)

    # Draw auxiliary lines for the theoretical main lobe width (±1.023 MHz)
    plt.axvline(x=fc / 1e6, color="black", linestyle="--", alpha=0.7, label="Chip Rate ($\pm$1.023 MHz)")
    plt.axvline(x=-fc / 1e6, color="black", linestyle="--", alpha=0.7)

    plt.title("Frequency Domain: GPS C/A CDMA Spectrum")
    plt.xlabel("Frequency Offset from Carrier (MHz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.grid(True)
    plt.xlim(-3, 3)  # Range where the main lobe and the first sidelobe are visible
    plt.ylim(-40, 5)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_gps_ca_spectrum()
