"""
Plot Pseudorange and carrier phase from RINEX observation file.

"""
import georinex as gr

from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)



infile = './3019148c.23o'

rnxobs = gr.load(infile)
print(rnxobs)

import matplotlib.pyplot as plt

ax = plt.gca()
satname = 'G14'
ax.plot(rnxobs.time, rnxobs['C1C'].sel(sv=satname))
ax.set_xlabel('GPS time')
ax.set_ylabel('Pseudorange [m]')
ax.grid(True)

fig, axes = plt.subplots(4,1, figsize=(10,8), sharex=True)
axes[0].plot(rnxobs.time, rnxobs['C1C'].sel(sv=satname), label=satname)
axes[1].plot(rnxobs.time, rnxobs['L1C'].sel(sv=satname), label=satname)
axes[2].plot(rnxobs.time, rnxobs['D1C'].sel(sv=satname), label=satname)
axes[3].plot(rnxobs.time, rnxobs['S1C'].sel(sv=satname), label=satname)

axes[0].set_title('Pseudo range[m]')
axes[0].set_ylabel(r'$\rho$ [m]')
axes[1].set_title('carrier phase')
axes[1].set_ylabel(r'$\phi$ [cycle]')
axes[2].set_title('Doppler frequency')
axes[2].set_ylabel('D[Hz]')
axes[3].set_title('C/N0')
axes[3].set_ylabel('S [dB]')
for ax in axes:
    ax.grid(True)

axes[3].set_xlabel('GPST')
plt.tight_layout()
plt.savefig('obs.png')

CLIGHT = 299792458.0
L1_FREQ=1.57542e9
wlen = CLIGHT / L1_FREQ
print(wlen)

# caluclate pseuro-range - carrier phase
cp_l1 = rnxobs['L1C'].sel(sv=satname)
pr_l1 = rnxobs['C1C'].sel(sv=satname)
dp_l1 = rnxobs['D1C'].sel(sv=satname)
sn_l1 = rnxobs['S1C'].sel(sv=satname)
cp_pr_l1 = cp_l1 - pr_l1 / wlen

# caluclate doppler - difference of carrier phase
b2 = cp_l1.diff('time') * (1/30.0) + dp_l1

fig, axes = plt.subplots(3,1, figsize=(12,8), sharex=True)
plt.suptitle(f'Analysis of Raw Measurement ({satname})')
axes[0].set_title(r'$-N - \frac{2}{\lambda}I$')
axes[0].plot(rnxobs.time, cp_pr_l1)
axes[0].set_ylabel('CP-PR')

axes[1].set_title(r'$\frac{1}{\lambda}(I + d/dt (I+T))$')
axes[1].plot(b2.time, b2)
axes[1].set_ylabel('DP - CP')
x = cp_pr_l1 + 2.0 * b2

axes[2].set_title(r'$N$')
axes[2].plot(x.time, -x.values)
axes[2].set_ylabel('bias [cycle]')

#axes[2].set_title('Compare Bias [cycle]')
#axes[2].plot(b2.time, -2.0*b2 + 2.0* int(b2.values[0]), label='Doppler and Carrier Phase')
#axes[2].plot(rnxobs.time, cp_pr_l1 - int(cp_pr_l1.values[0]), label='Pseudo range and Carrier Phase')
#axes[2].set_ylabel('Doppler - d/dt ADR')

for ax in axes:
    ax.grid(True)
axes[2].set_xlabel('GPST')
plt.savefig('bias.png')
plt.show()
