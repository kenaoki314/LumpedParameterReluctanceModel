"""Author Ken Aoki
Plotting graphs for Force vs displacement from adhesionPrediction.py"""





from adhesionPrediction import run_lpr, MAGNETS, D, X_MAX, IN_TO_M, N_TO_LBF, MU_0
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from adhesionPrediction  import (
    run_lpr,   
    MAGNETS,     
    D,           
    X_MAX,        
    IN_TO_M,      
    N_TO_LBF,     
    MU_0,         
)

def pick_magnet(): 
    print("\n" + "=" * 55)
    print("  LPR FORCE PREDICTOR — Select Magnet")
    print("=" * 55)
    for key, mag in MAGNETS.items():
        print(f"  [{key}]  {mag['label']}")
    while True:
        try:
            choice = int(input("\n  Enter magnet number: "))
            if choice in MAGNETS:
                return MAGNETS[choice]
            else:
                print(f"  Invalid. Please enter a number from the list above.")
        except ValueError:
            print("  Please enter a whole number.")


def pick_pole_width(magnet_params):
    """Print pole width menu, return the chosen width in inches."""
    print("\n" + "-" * 55)
    print(f"  Magnet: {magnet_params['label']}")
    print("  Available L-piece pole widths:")
    print("-" * 55)
 
    widths = magnet_params["pole_widths_in"]   
    for i, w in enumerate(widths):
        print(f"  [{i+1}]  w = {w}\"  ({w * IN_TO_M * 1e3:.2f} mm)")
 
    while True:
        try:
            choice = int(input("\n  Enter pole width number: "))
            if 1 <= choice <= len(widths):
                return widths[choice - 1]  
            else:
                print(f"  Invalid. Enter 1 to {len(widths)}.")
        except ValueError:
            print("  Please enter a whole number.")



def print_spot_check(data, magnet_params, w_in):
    """Print a formatted table of LPR predictions at key gap values."""
    CHECK_GAPS_MM = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0]   # [mm]
 
    print("\n" + "=" * 65)
    print(f"  LPR PREDICTIONS — {magnet_params['label']}")
    print(f"  w = {w_in}\"  |  mu_r = {data['mu_r']:.4f}  |  MMF = {data['MMF']:.1f} A")
    print(f"  NOTE: Single-branch MEC — leakage flux NOT modeled.")
    print("=" * 65)
    print(f"  {'x [mm]':>8}  {'F [N]':>10}  {'F [lbf]':>10}  {'kappa':>8}")
    print(f"  {'─'*46}")
 
    for x_mm in CHECK_GAPS_MM:
   
        idx = np.argmin(np.abs(data["x"] - x_mm * 1e-3))
        print(f"  {data['x'][idx]*1e3:>8.2f}  "
              f"{data['F'][idx]:>10.1f}  "
              f"{data['F'][idx]*N_TO_LBF:>10.2f}  "
              f"{data['kappa'][idx]:>8.3f}")
 
 

 
def make_plot(data, magnet_params, w_in):
    """
    Generate and save a two-panel figure:
      Left panel:  Force vs. air gap x [mm]    — dimensional
      Right panel: Force vs. kappa = x/w [—]   — validity envelope axis
    """
 

    color = "#1565C0" if "N52" in magnet_params["label"] else "#C62828"
 
    # Build figure title
    p = magnet_params
    mu_r_val = p["Br"] / (MU_0 * p["Hc"])
    title_str = (
        f"LPR Model — {p['label']}  |  Pole width w = {w_in}\"\n"
        f"μᵣ = {mu_r_val:.4f}  |  MMF = {data['MMF']:.1f} A  |  "
        f"Roters fringing  |  Leakage NOT modeled (LPR overpredicts)"
    )
 

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title_str, fontsize=10, fontweight="bold", y=1.02)
 
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.40)
    ax1 = fig.add_subplot(gs[0, 0])   # left:  F vs x
    ax2 = fig.add_subplot(gs[0, 1])   # right: F vs kappa
 
    ax1.plot(
        data["x"] * 1e3,   # m → mm
        data["F"],          # [N]
        color=color,
        linewidth=2.5,
        label=f'w = {w_in}"'
    )
 
    for x_mm in [1.0, 3.0, 5.0, 10.0]:
        idx   = np.argmin(np.abs(data["x"] - x_mm * 1e-3))
        F_val = data["F"][idx]
        ax1.annotate(
            f"x={x_mm:.0f}mm\n{F_val:.0f}N\n({F_val*N_TO_LBF:.1f}lbf)",
            xy=(data["x"][idx] * 1e3, F_val),
            xytext=(data["x"][idx] * 1e3 + 0.7, F_val + 20),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8),
        )
 
    ax1.set_xlabel("Air gap  x  [mm]", fontsize=12)
    ax1.set_ylabel("Adhesion force  F  [N]", fontsize=12)
    ax1.set_title("Force vs. Air Gap", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, X_MAX * 1e3)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
 
  
   
    ax1b = ax1.twinx()
    ax1b.set_ylim(np.array(ax1.get_ylim()) * N_TO_LBF)
    ax1b.set_ylabel("Force  [lbf]", fontsize=10, color="#888888")
    ax1b.tick_params(axis="y", labelcolor="#888888")
 
    ax2.plot(
        data["kappa"],   
        data["F"],      
        color=color,
        linewidth=2.5,
        label=f'w = {w_in}"'
    )
 

    ax2.axvline(x=0.1, color="#2196F3", linewidth=1.5, linestyle=":",
                label=r"$\kappa$ = 0.1")
    ax2.axvline(x=0.5, color="#888888", linewidth=1.2, linestyle="--",
                label=r"$\kappa$ = 0.5")
    ax2.axvline(x=1.0, color="#444444", linewidth=1.2, linestyle="-.",
                label=r"$\kappa$ = 1.0")
 
    ax2.set_xlabel(r"Dimensionless gap  $\kappa = x/w$  [—]", fontsize=12)
    ax2.set_ylabel("Adhesion force  F  [N]", fontsize=12)
    ax2.set_title(r"Force vs. $\kappa$  (validity envelope axis)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")
 

    ax2b = ax2.twinx()
    ax2b.set_ylim(np.array(ax2.get_ylim()) * N_TO_LBF)
    ax2b.set_ylabel("Force  [lbf]", fontsize=10, color="#888888")
    ax2b.tick_params(axis="y", labelcolor="#888888")
 

    fig.text(
        0.5, -0.04,
        f"Br = {p['Br']} T  |  Hc = {p['Hc']/1e3:.0f} kA/m  |  "
        f"μᵣ = {mu_r_val:.4f} (Br/μ₀Hc)  |  "
        f"l_m = {p['l_m']*1e3:.1f} mm  |  d = {D*1e3:.1f} mm  |  "
        f"w = {w_in * IN_TO_M * 1e3:.1f} mm\n"
        "WARNING: Grade-average Br/Hc used (IEC 60404-8-1). "
        "Leakage flux NOT modeled — LPR overpredicts for tall U-channel geometry.",
        ha="center", fontsize=8, color="#666666", style="italic"
    )
 
    magnet_short = p["label"].split("|")[0].strip().replace(" ", "_")
    filename = f"lpr_{magnet_short}_w{w_in}in.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {filename}")
    plt.show()
 
 

 
if __name__ == "__main__":
 
    print("\n  LPR Magnetic Adhesion Force Predictor")
    print("  RIP Lab — University of Hawaiʻi at Mānoa")
    print("  Physics: lpr_calculations.py")
    print("  WARNING: Single-branch MEC. Leakage not modeled.")
 
    chosen_magnet = pick_magnet()
    chosen_width  = pick_pole_width(chosen_magnet)
 
    print(f"\n  Running LPR sweep...")
    data = run_lpr(chosen_magnet, chosen_width)
    print(f"  Done.")
 
    print_spot_check(data, chosen_magnet, chosen_width)
    make_plot(data, chosen_magnet, chosen_width)