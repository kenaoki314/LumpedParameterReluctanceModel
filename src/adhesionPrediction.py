# ============================================================
# SCRIPT NAME: lpr_calculations.py
# PURPOSE: Define all LPR model parameters, functions, and
#          run the force vs air gap sweep for all 6 magnet +
#          L-piece configurations. No plotting — output only.
# AUTHOR: Ken Aoki
# DATE: 2026-06-27
# LAB: RIP Lab, University of Hawaii at Manoa
#
# HOW TO USE:
#   Run standalone:  python lpr_calculations.py
#     → prints console tables for all 6 configurations
#
#   Import into another script (e.g. lpr_plots.py):
#     from lpr_calculations import run_all_configs, MAGNETS, X_MAX, N_TO_LBF
#     results = run_all_configs()
# ============================================================

# --- IMPORTS ---
# numpy is the only dependency here — no matplotlib needed
import numpy as np    # NumPy: matrix math (equivalent to MATLAB built-ins)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: CONSTANTS
# (module-level, available to any script that imports this file)
# ─────────────────────────────────────────────────────────────────────────────
MU_0     = 4 * np.pi * 1e-7   # Permeability of free space [H/m]
IN_TO_M  = 0.0254              # Unit conversion: 1 inch = 0.0254 m
N_TO_LBF = 1.0 / 4.44822      # Unit conversion: Newtons → pounds-force

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MAGNET PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY NOTE:
#   Both magnets are magnetized through their THICKNESS (shortest dimension).
#   l_m = path length through magnet along magnetization direction [m]
#   A_m = magnet cross-sectional area perpendicular to B [m^2]
#
#   N52 (0.5 × 0.5 × 2 in), magnetized through 0.5" thickness:
#       l_m = 0.5"  →  0.0127 m
#       A_m = 2" × 0.5" face  →  6.45e-4 m^2
#
#   N48 (1 × 1 × 2 in), magnetized through 1" thickness:
#       l_m = 1.0"  →  0.0254 m
#       A_m = 2" × 1.0" face  →  1.29e-3 m^2
#
# MATERIAL NOTE:
#   Hc uses grade-average values from IEC 60404-8-1.
#   TODO: Replace with exact supplier values when obtained from CMS and Apex.

MAGNETS = {

    "N52  |  0.5×0.5×2 in  |  CMS Magnetics": {
        "l_m"            : 0.5 * IN_TO_M,                        # magnet thickness [m]
        "A_m"            : (2.0 * IN_TO_M) * (0.5 * IN_TO_M),   # magnet face area [m^2]
        "Br"             : 1.44,      # Remanence [T]       — CMS Magnetics datasheet
        "Hc"             : 900e3,     # Coercivity [A/m]    — N52 grade avg (IEC 60404-8-1)
        "mu_r"           : 1.05,      # Recoil permeability — typical sintered NdFeB
        "pole_widths_in" : [0.25, 0.50, 1.00],   # L-piece widths tested [inches]
    },

    "N48  |  1×1×2 in  |  Apex Magnetics": {
        "l_m"            : 1.0 * IN_TO_M,                        # magnet thickness [m]
        "A_m"            : (2.0 * IN_TO_M) * (1.0 * IN_TO_M),   # magnet face area [m^2]
        "Br"             : 1.40,      # Remanence [T]       — N48 grade avg (IEC 60404-8-1)
        "Hc"             : 860e3,     # Coercivity [A/m]    — N48 grade avg (IEC 60404-8-1)
        "mu_r"           : 1.05,      # Recoil permeability — typical sintered NdFeB
        "pole_widths_in" : [0.50, 1.00, 1.50],   # L-piece widths tested [inches]
    },

}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SHARED GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────
D      = 2.0 * IN_TO_M    # Pole face depth [m] — 8 L-pieces × 0.25 in = 2.0 in (constant)
X_MIN  = 0.10e-3          # Minimum air gap [m] — 0.1 mm, avoids divide-by-zero
X_MAX  = 15.0e-3          # Maximum air gap [m] — 15 mm sweep ceiling
N_PTS  = 500              # Number of points in gap sweep

# Air gap array [m] — same as MATLAB: linspace(X_MIN, X_MAX, N_PTS)
X_ARR  = np.linspace(X_MIN, X_MAX, N_PTS)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: LPR MODEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def magnet_reluctance(l_m, mu_r, A_m):
    """
    Reluctance of the permanent magnet [A/Wb].

    Formula:  R_m = l_m / (mu_0 * mu_r * A_m)

    This is the 'source resistance' in the magnetic circuit.
    Uses the magnet's own face area A_m — NOT the pole face area.

    Args:
        l_m   : magnet path length along magnetization direction [m]
        mu_r  : recoil permeability of magnet [-]
        A_m   : magnet cross-sectional area perpendicular to B [m^2]
    Returns:
        R_m   : magnet reluctance [A/Wb]
    """
    return l_m / (MU_0 * mu_r * A_m)


def effective_gap_area(x, w, d):
    """
    Effective air gap area using Roters fringing correction [m^2].

    Formula:  A_eff = (w + 2x)(d + 2x)

    Fringing flux escapes around all four edges of the pole face,
    making the effective area larger than the physical pole face (w × d).
    The 2x factor accounts for fringing on both sides of each edge.

    Args:
        x   : air gap distance [m]
        w   : pole face width [m]
        d   : pole face depth [m]
    Returns:
        A_eff : effective gap area [m^2]
    """
    return (w + 2 * x) * (d + 2 * x)


def gap_reluctance_total(x, A_eff):
    """
    Total reluctance of BOTH air gaps in series [A/Wb].

    Formula:  R_g_total = (2 * x) / (mu_0 * A_eff)

    Two gaps in series (N-pole gap + S-pole gap), each with
    reluctance x/(mu_0 * A_eff), giving a total of (2x)/(mu_0 * A_eff).

    Args:
        x     : air gap at each pole face [m]
        A_eff : effective gap area [m^2]
    Returns:
        R_g_total : total air gap reluctance [A/Wb]
    """
    return (2 * x) / (MU_0 * A_eff)


def compute_flux(MMF, R_m, R_g_total):
    """
    Magnetic flux through the circuit [Wb].

    Formula:  Phi = MMF / (R_m + R_g_total)

    Ohm's law analog: Phi = MMF / R_total
    (same logic as I = V / R in electric circuits)

    Args:
        MMF       : magnetomotive force [A]  (= Hc * l_m)
        R_m       : magnet reluctance [A/Wb]
        R_g_total : total gap reluctance [A/Wb]
    Returns:
        Phi : magnetic flux [Wb]
    """
    return MMF / (R_m + R_g_total)


def adhesion_force(Phi, A_eff):
    """
    Total adhesion force across BOTH pole faces [N].

    Formula:  F = Phi^2 / (mu_0 * A_eff)

    Derived from Maxwell stress tensor at each gap:
        F_one_gap = Phi^2 / (2 * mu_0 * A_eff)
    Two gaps contribute equally:
        F_total   = 2 * F_one = Phi^2 / (mu_0 * A_eff)

    Args:
        Phi   : magnetic flux through circuit [Wb]
        A_eff : effective gap area [m^2]
    Returns:
        F : total adhesion force [N]
    """
    return Phi**2 / (MU_0 * A_eff)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MAIN COMPUTATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_all_configs():
    """
    Run the LPR force sweep for all magnet + L-piece configurations.

    Returns a nested dictionary:
        results[magnet_name][config_label] = {
            'x'     : air gap array [m],          shape (N_PTS,)
            'F'     : adhesion force array [N],    shape (N_PTS,)
            'kappa' : dimensionless ratio x/w,     shape (N_PTS,)
            'A_eff' : effective gap area [m^2],    shape (N_PTS,)
            'w'     : pole face width [m],         scalar
            'w_in'  : pole face width [inches],    scalar
        }

    This dict is imported by lpr_plots.py for graphing.
    """
    results = {}

    for magnet_name, p in MAGNETS.items():

        # MMF = Hc * l_m  [A]  (magnetomotive force — the 'voltage' driving the circuit)
        MMF = p["Hc"] * p["l_m"]

        # R_m is fixed for a given magnet — does not change with L-piece config
        R_m = magnet_reluctance(p["l_m"], p["mu_r"], p["A_m"])

        results[magnet_name] = {}

        for w_in in p["pole_widths_in"]:

            w     = w_in * IN_TO_M          # pole face width [m]
            label = f'w = {w_in}" ({w*1e3:.2f} mm)'

            # Pre-allocate output arrays (like zeros() in MATLAB)
            F_arr     = np.zeros(N_PTS)
            A_eff_arr = np.zeros(N_PTS)

            # Sweep: compute force at each air gap value
            # In MATLAB this would be a vectorized operation;
            # in Python we loop explicitly for clarity.
            for i, x in enumerate(X_ARR):
                A_eff        = effective_gap_area(x, w, D)
                R_g          = gap_reluctance_total(x, A_eff)
                Phi          = compute_flux(MMF, R_m, R_g)
                F_arr[i]     = adhesion_force(Phi, A_eff)
                A_eff_arr[i] = A_eff

            results[magnet_name][label] = {
                "x"     : X_ARR.copy(),       # air gap [m]
                "F"     : F_arr,              # adhesion force [N]
                "kappa" : X_ARR / w,          # dimensionless gap ratio κ = x/w
                "A_eff" : A_eff_arr,          # effective area [m^2]
                "w"     : w,                  # pole width [m]
                "w_in"  : w_in,               # pole width [in]
            }

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CONSOLE OUTPUT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def print_tables(results):
    """
    Print a formatted summary table for each configuration.
    Shows F [N], F [lbf], kappa, and A_eff at 5 spot-check air gap values.
    """
    CHECK_GAPS = [0.5e-3, 1.0e-3, 2.0e-3, 5.0e-3, 10.0e-3]   # spot-check x values [m]

    print("=" * 70)
    print("LPR MODEL — ADHESION FORCE PREDICTIONS")
    print("=" * 70)

    for magnet_name, p in MAGNETS.items():
        MMF = p["Hc"] * p["l_m"]
        R_m = magnet_reluctance(p["l_m"], p["mu_r"], p["A_m"])

        print(f"\n{'─'*70}")
        print(f"  {magnet_name}")
        print(f"{'─'*70}")
        print(f"  l_m   = {p['l_m']*1e3:.3f} mm")
        print(f"  A_m   = {p['A_m']*1e6:.3f} mm^2")
        print(f"  Br    = {p['Br']:.3f} T")
        print(f"  Hc    = {p['Hc']/1e3:.0f} kA/m")
        print(f"  mu_r  = {p['mu_r']:.2f}")
        print(f"  MMF   = Hc x l_m = {MMF:.2f} A")
        print(f"  R_m   = {R_m:.3e} A/Wb")
        print(f"  d     = {D*1e3:.1f} mm  (pole face depth, constant)")

        for label, data in results[magnet_name].items():
            w    = data["w"]
            w_in = data["w_in"]
            print(f"\n    Config: {label}")
            print(f"    Pole face area (w x d) = {w * D * 1e6:.2f} mm^2")
            print(f"    {'x [mm]':>8}  {'F [N]':>9}  {'F [lbf]':>9}  "
                  f"{'kappa':>9}  {'A_eff [mm^2]':>14}")
            print(f"    {'─'*55}")

            for x_chk in CHECK_GAPS:
                if x_chk <= X_MAX:
                    # np.argmin finds the index of the closest x in X_ARR
                    # equivalent to MATLAB: [~, idx] = min(abs(X_ARR - x_chk))
                    idx = np.argmin(np.abs(data["x"] - x_chk))
                    print(f"    {data['x'][idx]*1e3:>8.2f}  "
                          f"{data['F'][idx]:>9.2f}  "
                          f"{data['F'][idx]*N_TO_LBF:>9.3f}  "
                          f"{data['kappa'][idx]:>9.3f}  "
                          f"{data['A_eff'][idx]*1e6:>14.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: STANDALONE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: The block below ONLY runs when you execute this file directly:
#           python lpr_calculations.py
#
#       It does NOT run when lpr_plots.py imports this file.
#       This is Python's standard way of separating "library code"
#       from "run this now" code.
#       MATLAB equivalent: putting code outside of functions in a script.

if __name__ == "__main__":
    results = run_all_configs()
    print_tables(results)