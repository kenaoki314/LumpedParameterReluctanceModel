"""
Multi-curve F(x, w) plot: adhesion force vs airgap for several square pole widths.
Demonstrates the Roters fringing correction — A_eff grows with gap, so the
force rolls off more gradually than a constant-area model would predict.
NdFeB N52 material parameters throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from adhesionPrediction import (
    internal_reluctance, total_airgap_reluctance,
    flux_calculation, adhesion_force
)

# --- Fixed material parameters (NdFeB N52) ---
mu_0 = 4 * np.pi * 1e-7   # permeability of free space [H/m]
mu_r = 1.05                # relative permeability of NdFeB
l_m  = 0.010               # magnet length in magnetization direction [m]
H_c  = 900_000             # coercive field [A/m]
MMF  = H_c * l_m           # magnetomotive force [A·t]

# --- Pole widths to compare (square poles: w = d) ---
pole_widths_mm = [10, 15, 20, 25, 30]

# --- Airgap sweep ---
airgap_mm = np.linspace(0.05, 10.0, 500)
airgap_m  = airgap_mm * 1e-3

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pole_widths_mm)))

fig, ax = plt.subplots(figsize=(9, 5))

for w_mm, color in zip(pole_widths_mm, colors):
    w = w_mm * 1e-3   # square pole face: d = w
    d = w

    R_m = internal_reluctance(l_m, mu_0, mu_r, w, d)

    forces = np.array([
        adhesion_force(
            flux_calculation(total_airgap_reluctance(x, mu_0, w, d), R_m, MMF),
            mu_0, x, w, d
        )
        for x in airgap_m
    ])

    ax.plot(airgap_mm, forces, color=color, linewidth=2, label=f"w = d = {w_mm} mm")

ax.set_xlabel("Air Gap [mm]", fontsize=13)
ax.set_ylabel("Adhesion Force [N]", fontsize=13)
ax.set_title(
    r"LPR Model — $F(x,\,w)$ with Roters Fringing Correction"
    "\n"
    r"NdFeB N52, $l_m$ = 10 mm, $H_c$ = 900 kA/m, square pole faces",
    fontsize=12
)
ax.legend(title="Pole size", fontsize=10, title_fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "..", "adhesion_force_curve.png"),
    dpi=150
)
plt.show()
print("Plot saved to adhesion_force_curve.png")
