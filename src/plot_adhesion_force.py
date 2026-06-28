# ============================================================
# SCRIPT NAME: lpr_plots.py
# PURPOSE: Generate all plots for the LPR model force predictions.
#          Imports computed results from lpr_calculations.py.
#          Does NOT redefine any physics — only graphing here.
# AUTHOR: Ken Aoki
# DATE: 2026-06-27
# LAB: RIP Lab, University of Hawaii at Manoa
#
# REQUIRES: lpr_calculations.py must be in the same folder.
#
# HOW TO RUN:
#   python lpr_plots.py
#   → produces and saves lpr_force_curves.png
# ============================================================

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import everything needed from the calculations file.
# Python runs lpr_calculations.py once when this line executes,
# but because of the "if __name__ == '__main__'" guard in that file,
# it does NOT print any tables — it just loads the functions and constants.
from lpr_calculations import (
    run_all_configs,   # function that runs the full sweep
    MAGNETS,           # magnet parameter dict (used for labels)
    X_MAX,             # max air gap [m] (used to set axis limits)
    N_TO_LBF,          # unit conversion factor N → lbf
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: RUN CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
# Call the computation function from lpr_calculations.py.
# This gives us a nested dict with x, F, kappa arrays for each config.
results = run_all_configs()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: PLOT STYLING
# ─────────────────────────────────────────────────────────────────────────────
# Colors: 3 shades per magnet (dark → light as pole width increases)
N52_COLORS = ["#1565C0", "#1E88E5", "#90CAF9"]   # blue family (N52)
N48_COLORS = ["#B71C1C", "#E53935", "#EF9A9A"]   # red family (N48)
LINESTYLES = ["-", "--", "-."]                    # solid, dashed, dash-dot

COLOR_SETS = [N52_COLORS, N48_COLORS]   # index matches order of MAGNETS dict

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: BUILD FIGURE
# ─────────────────────────────────────────────────────────────────────────────
# Layout: 2 rows × 2 columns
#   Row 1 (top):    Force [N] vs air gap x [mm]   — dimensional view
#   Row 2 (bottom): Force [N] vs kappa = x/w      — dimensionless view (paper axis)

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "LPR Model — Magnetic Adhesion Force Predictions\n"
    "Roters fringing correction  |  Steel reluctance neglected  |  Both pole faces included",
    fontsize=13,
    fontweight="bold",
    y=0.98
)

# gridspec controls subplot spacing
# hspace = vertical gap between rows, wspace = horizontal gap between columns
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32)

# Create the four subplot axes
ax_N52_Fx  = fig.add_subplot(gs[0, 0])   # top-left:     N52, F vs x
ax_N48_Fx  = fig.add_subplot(gs[0, 1])   # top-right:    N48, F vs x
ax_N52_Fk  = fig.add_subplot(gs[1, 0])   # bottom-left:  N52, F vs kappa
ax_N48_Fk  = fig.add_subplot(gs[1, 1])   # bottom-right: N48, F vs kappa

# Group axes by magnet so we can loop
# Each tuple: (F-vs-x axis, F-vs-kappa axis)
axes_pairs = [
    (ax_N52_Fx, ax_N52_Fk),
    (ax_N48_Fx, ax_N48_Fk),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: PLOT EACH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# enumerate gives us both an index and the value in each loop iteration.
# MATLAB equivalent: for col = 1:length(magnet_names)

for col, (magnet_name, configs) in enumerate(results.items()):

    colors  = COLOR_SETS[col]
    ax_Fx   = axes_pairs[col][0]   # F vs x panel for this magnet
    ax_Fk   = axes_pairs[col][1]   # F vs kappa panel for this magnet

    for c_idx, (label, data) in enumerate(configs.items()):

        clr  = colors[c_idx]
        ls   = LINESTYLES[c_idx]

        # ── Plot 1: Force vs air gap (dimensional) ───────────────────────
        ax_Fx.plot(
            data["x"] * 1e3,    # convert m → mm for x-axis
            data["F"],          # force in Newtons
            color=clr,
            linestyle=ls,
            linewidth=2.0,
            label=label
        )

        # ── Plot 2: Force vs kappa (dimensionless) ───────────────────────
        ax_Fk.plot(
            data["kappa"],      # kappa = x/w (dimensionless)
            data["F"],          # force in Newtons
            color=clr,
            linestyle=ls,
            linewidth=2.0,
            label=f'w = {data["w_in"]}"'
        )

    # ── FORMAT: F vs x panel ─────────────────────────────────────────────
    ax_Fx.set_xlabel("Air gap  x  [mm]", fontsize=11)
    ax_Fx.set_ylabel("Adhesion force  F  [N]", fontsize=11)
    ax_Fx.set_title(magnet_name, fontsize=11, fontweight="bold")
    ax_Fx.legend(fontsize=9, title="L-piece width", title_fontsize=9)
    ax_Fx.grid(True, alpha=0.3)
    ax_Fx.set_xlim(0, X_MAX * 1e3)
    ax_Fx.set_ylim(bottom=0)

    # Secondary y-axis in lbf (twin of the N axis)
    ax_Fx2 = ax_Fx.twinx()
    ax_Fx2.set_ylim(np.array(ax_Fx.get_ylim()) * N_TO_LBF)
    ax_Fx2.set_ylabel("Force [lbf]", fontsize=9, color="#555555")
    ax_Fx2.tick_params(axis="y", labelcolor="#555555")

    # ── FORMAT: F vs kappa panel ─────────────────────────────────────────
    ax_Fk.set_xlabel(r"Dimensionless air gap   $\kappa = x/w$   [—]", fontsize=11)
    ax_Fk.set_ylabel("Adhesion force  F  [N]", fontsize=11)
    ax_Fk.set_title(
        f"{magnet_name}\n(κ view — validity envelope axis)",
        fontsize=10,
        fontweight="bold"
    )
    ax_Fk.legend(fontsize=9, title="L-piece width", title_fontsize=9)
    ax_Fk.grid(True, alpha=0.3)
    ax_Fk.set_xlim(left=0)
    ax_Fk.set_ylim(bottom=0)

    # Vertical reference lines at candidate validity bounds κ = 0.5 and 1.0
    # These are the hypothesized accuracy breakpoints — paper will test these
    ax_Fk.axvline(x=0.5, color="#888888", linewidth=1.2,
                  linestyle=":",  label="κ = 0.5")
    ax_Fk.axvline(x=1.0, color="#555555", linewidth=1.2,
                  linestyle="--", label="κ = 1.0")
    ax_Fk.legend(fontsize=9, title="L-piece width", title_fontsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: FOOTNOTE AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    "NOTE: Hc uses N-grade average (IEC 60404-8-1) — replace with supplier values when obtained. "
    "Fringing: Roters A_eff = (w+2x)(d+2x). Steel reluctance neglected.",
    ha="center",
    fontsize=8,
    color="#666666",
    style="italic"
)

plt.savefig("lpr_force_curves.png", dpi=150, bbox_inches="tight")
print("Plot saved → lpr_force_curves.png")
plt.show()