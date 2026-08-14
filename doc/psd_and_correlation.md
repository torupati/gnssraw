# Power Spectral Density and Correlation of GNSS signals

## GPS C/A L1 code

Basic signal structure is composed from carrier (L1 band, 1.5GHz) and spreading code (1.023MHz, chip rate, 1023 chip) and data (50 bps).

Multiplied the $C(t)D(t)$ with carrier wave $\cos(2\pi f_c t)$, its theoretical PSD is PSD of 1.023MHz rectified wave form. The shape is Fig.1.

Chip period $T_d = \frac{1}{f_c}$

$$
g(t) = \mathbf{rect}_{T_d} (t)
$$

$$
\Phi(f) = T_d \mathbf{sinc}(\pi fT_d)
$$


$$
G(f) = \frac{\sin^2(\frac{\pi f}{f_c})}{\left(\frac{πf}{fc}\right)^2}
$$

![GPS C/A PSD and ACF](./figures/gps_ca_theoretical_psd_bw8.0MHz.png)

**Left — PSD.**
The spectrum is $\text{sinc}^2$-shaped, centered at $f = 0$.
The main lobe spans $[-f_c,\, +f_c]$ (null-to-null bandwidth $= 2f_c = 2.046\ \text{MHz}$).
Side lobes roll off as $1/f^2$, falling to approximately $-13\ \text{dB}$ at the first side lobe.
In a real receiver the front-end filter limits the captured bandwidth; a typical 2 MHz double-sided filter retains mostly the main lobe.

**Right — Autocorrelation function (ACF).**
The ACF of a rectangular chip pulse is the convolution of the pulse with itself, which gives a triangle:

$$
R(\tau) = \max\!\left(1 - \frac{|\tau|}{T_c},\; 0\right)
$$

It peaks at $\tau = 0$ and reaches zero at $|\tau| = T_c \approx 0.977\ \mu\text{s}$.
The constant slope $1/T_c$ within $|\tau| < T_c$ is exploited by the DLL (Delay Lock Loop):
the Early-minus-Late discriminator measures the slope asymmetry to estimate the tracking error.
Narrowing the correlator spacing (e.g., narrow correlator, strobe) improves noise immunity
and multipath resistance without changing the underlying triangular shape.

## BOC

BOC (Binary Offset Carrier) places a square-wave subcarrier of frequency $f_s$ on top of
the BPSK spreading code.
The chip waveform alternates sign every half-subcarrier period, splitting the spectral power
away from DC into two lobes centred at $\pm f_s$.
For BOC(1,1), $f_c = f_s = 1.023\ \text{MHz}$ and the subcarrier completes one full cycle
per chip ($N = 2$ half-cycles per chip).

The PSD formula for even $N$ is:

$$
G_\text{BOC}(f) = f_c \left(\frac{\sin\!\left(\frac{\pi f}{f_c}\right)\sin\!\left(\frac{\pi f}{2f_s}\right)}{\pi f}\right)^{\!2} \tan^2\!\left(\frac{\pi f}{2f_s}\right)
$$

![BPSK vs BOC(1,1) PSD and ACF](./figures/gps_boc_theoretical_psd_bw8.0MHz.png)

Both curves are plotted at **equal total power**: each PSD is normalised so that its integral
over all frequencies equals 1, and the reference (0 dB) is the BPSK peak at $f = 0$.

**Left — PSD.**
The BPSK/C/A spectrum (blue dashed) has its peak at $f = 0$.
The BOC(1,1) spectrum (red solid) has two main lobes centred at $\pm f_s = \pm 1.023\ \text{MHz}$
and a deep null at $f = 0$.
Each BOC lobe carries roughly half the total power in a bandwidth similar to the BPSK main lobe,
so its peak spectral density is approximately 3 dB lower (visible in the figure as ≈ −2 dB).
The frequency separation from DC improves isolation from low-frequency interference
and from GPS L1 C/A on the same band, which is the motivation for using BOC in Galileo E1
and GPS L1C.

**Right — ACF.**
The BOC(1,1) autocorrelation (red solid) is piecewise linear with three segments per side:

$$
R_\text{BOC}(\tau) = \begin{cases}
1 - 3|\tau|/T_c & |\tau| \le T_c/2 \\
-1 + |\tau|/T_c & T_c/2 < |\tau| \le T_c \\
0 & |\tau| > T_c
\end{cases}
$$

Key features compared with BPSK:

- **Steeper central slope** ($3/T_c$ vs $1/T_c$): a narrower main peak, which makes the DLL
  discriminator more sensitive to small timing errors and improves range measurement precision.
- **Side lobes at $\pm T_c/2$ (value $\approx -0.5$)**: these are secondary maxima that can cause
  false lock in a standard DLL if the initial code phase is off by $\pm T_c/2$.
  Receiver designs for BOC must use an unambiguous acquisition strategy (e.g., BPSK-like
  envelope, bump-jumping, or MBOC modifications) to avoid locking onto a side lobe.
