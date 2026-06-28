# test test test sahur
"""
Author: Ken Aoki. 6/26/26
Lumped Parameter Reluctance (LPR) model for adhesion force prediction.
Test article: bar magnet with L-shaped iron pole pieces on steel plate.

Magnetic circuit (series):
    MMF source (magnet) → magnet reluctance R_m → two air gaps 2*R_g → steel (neglected)

Fringing correction (Roters):
    A_eff = (w + 2*x) * (d + 2*x)
    where w = pole face width, d = pole face depth, x = air gap
"""

import numpy as np


def internal_reluctance(l_m: float, mu_0: float, mu_r: float, A_m: float) -> float:
    """
    Reluctance of the permanent magnet.

    Args:
        l_m   : magnet length along magnetization direction [m]
        mu_0  : permeability of free space [H/m]
        mu_r  : relative recoil permeability of magnet [-]
        A_m   : magnet cross-sectional area [m^2]

    Returns:
        R_m   : magnet reluctance [A/Wb]
    """
    R_m = l_m / (mu_0 * mu_r * A_m)
    return R_m


def effective_gap_area(x: float, w: float, d: float) -> float:
    """
    Effective air gap area using Roters fringing correction.
    Fringing flux spreads beyond the physical pole edge, increasing
    the effective cross-sectional area of each gap.

    Args:
        x : air gap distance [m]
        w : pole face width [m]
        d : pole face depth (out-of-plane) [m]

    Returns:
        A_eff : effective gap area [m^2]

    Note:
        Roters:    A_eff = (w + 2*x) * (d + 2*x)   <-- used here
        Simpler:   A_eff = (w + x)   * (d + x)      <-- uncomment to compare
    """
    A_eff = (w + 2 * x) * (d + 2 * x)
    # A_eff = (w + x) * (d + x)   # alternative — compare against data to decide
    return A_eff


def total_gap_reluctance(x: float, mu_0: float, A_eff: float) -> float:
    """
    Total reluctance of BOTH air gaps combined (two gaps in series,
    one at each pole face: N-pole gap + S-pole gap).

    Args:
        x     : air gap at each pole face [m]
        mu_0  : permeability of free space [H/m]
        A_eff : effective gap cross-sectional area [m^2]

    Returns:
        R_g_total : total air gap reluctance [A/Wb]
    """
    R_g_total = (2 * x) / (mu_0 * A_eff)
    return R_g_total


def flux_calculation(R_m: float, R_g_total: float, MMF: float) -> float:
    """
    Magnetic flux via circuit Ohm's law analog: Phi = MMF / R_total.
    Steel reluctance neglected (mu_steel >> mu_0).

    Args:
        R_m       : magnet reluctance [A/Wb]
        R_g_total : total air gap reluctance [A/Wb]
        MMF       : magnetomotive force [A]  (= Hc * l_m)

    Returns:
        Phi : magnetic flux [Wb]
    """
    R_total = R_m + R_g_total
    Phi = MMF / R_total
    return Phi


def adhesion_force(Phi: float, mu_0: float, A_eff: float) -> float:
    """
    Total adhesion force across both pole faces via Maxwell stress.

    Each gap contributes F_one = Phi^2 / (2 * mu_0 * A_eff).
    Two gaps in the circuit → total = 2 * F_one = Phi^2 / (mu_0 * A_eff).

    Args:
        Phi   : magnetic flux through the circuit [Wb]
        mu_0  : permeability of free space [H/m]
        A_eff : effective gap area [m^2]

    Returns:
        F : total adhesion force [N]
    """
    F = Phi**2 / (mu_0 * A_eff)
    return F