import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Earth parameters
R_earth = 6371.0          # km
GM = 3.986004418e14       # m^3/s^2

# Kepler's third law: T = 2*pi * sqrt(r^3 / GM)
def kepler_period_h(r_km):
    r_m = r_km * 1e3
    T_s = 2 * np.pi * np.sqrt(r_m**3 / GM)
    return T_s / 3600  # hours

# Continuous curve: r from 200 km altitude to 50000 km altitude
r_alt_cont = np.linspace(200, 50000, 2000)
r_cont = R_earth + r_alt_cont
T_cont = kepler_period_h(r_cont)

# Satellite data: (name, altitude_km, period_h, color)
satellites = [
    ("ISS",        400,    92/60,   "#888888"),
    ("Starlink",   550,    92.4/60, "#e57373"),
    ("GLONASS",    19100,  11.25,   "#64b5f6"),
    ("GPS",        20200,  12.0,    "#1565c0"),
    ("BeiDou MEO", 21500,  12.0,    "#ff8f00"),
    ("Galileo",    23220,  14.0,    "#43a047"),
    ("GEO (QZSS/IRNSS/BeiDou)", 35786, 24.0, "#ab47bc"),
]

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(r_cont, T_cont, color="#1976d2", linewidth=2,
        label=r"Kepler: $T = 2\pi\,\sqrt{r^3/GM}$")

for name, alt, T_actual, color in satellites:
    r = R_earth + alt
    T_kepler = kepler_period_h(r)
    ax.scatter(r, T_kepler, color=color, s=80, zorder=5)
    # label offset
    offset_x = 200
    offset_y = 0.3
    if name == "ISS":
        offset_x = 200; offset_y = 0.6
    elif name == "Starlink":
        offset_x = 200; offset_y = -1.0
    elif "GEO" in name:
        offset_x = -6000; offset_y = 0.5
    ax.annotate(name, xy=(r, T_kepler),
                xytext=(r + offset_x + 3000, T_kepler + offset_y),
                fontsize=8.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

# Annotate GEO altitude line
ax.axvline(R_earth + 35786, color="#ab47bc", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_xlabel("Orbital radius $r$ (km)", fontsize=11)
ax.set_ylabel("Orbital period $T$ (hours)", fontsize=11)
ax.set_title("Kepler's Third Law — Orbital Radius vs. Period\n"
             r"$T^2 \propto r^3$  ($T = 2\pi\sqrt{r^3/GM_\oplus}$)", fontsize=12)
ax.legend(fontsize=9)
ax.set_yticks([6, 12, 18, 24])
ax.set_yticklabels(["6 h", "12 h", "18 h", "24 h"])
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 48000)
ax.set_ylim(0, 28)

# Mark Earth's surface (r = R_earth) on the x-axis
ax.axvline(R_earth, color="#4caf50", linewidth=1.2, linestyle=":", alpha=0.8)
ax.text(R_earth + 120, 1.0, f"$R_\\oplus$ = {R_earth:.0f} km",
        color="#4caf50", fontsize=8, va="bottom")

# Secondary x-axis for altitude
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
alt_ticks = [0, 550, 10000, 20200, 35786]
ax2.set_xticks([R_earth + a for a in alt_ticks])
ax2.set_xticklabels([f"{a:,}" for a in alt_ticks], fontsize=8)
ax2.set_xlabel("Altitude above Earth's surface (km)", fontsize=9)

plt.tight_layout()
plt.savefig("doc/figures/kepler_third_law.png", dpi=150, bbox_inches="tight")
print("Saved: doc/figures/kepler_third_law.png")
