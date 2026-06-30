"""author Ken Aoki 
Lumped Parameter Reluctance Calculation """

import numpy as np 
import math
import matplotlib
import matplotlib.gridspec as gridspec 

MU_0 = 4 * math.pi * 1e-7
IN_TO_M = 0.0254
N_TO_LBF = 1 / 4.44822 

X_MIN = 0.1e-3
X_MAX = 15e-3 
N_PTS = 500 
X_ARR = np.linspace(X_MIN, X_MAX, N_PTS) 

MAGNETS = {
    1: {
        "label"          : "N52  |  0.5×0.5×2 in  |  CMS Magnetics",
        "l_m"            : 0.5 * IN_TO_M,                        # magnet thickness [m]
        "A_m"            : (2.0 * IN_TO_M) * (0.5 * IN_TO_M),   # magnet face area [m²]
        "Br"             : 1.44,      # Remanence [T]
        "Hc"             : 900e3,     # Coercivity [A/m]
        "pole_widths_in" : [0.25, 0.50, 1.00],   # available L-piece widths [in]
    },
    2: {
        "label"          : "N48  |  1×1×2 in  |  Apex Magnetics",
        "l_m"            : 1.0 * IN_TO_M,
        "A_m"            : (2.0 * IN_TO_M) * (1.0 * IN_TO_M),
        "Br"             : 1.40,
        "Hc"             : 860e3,
        "pole_widths_in" : [0.50, 1.00, 1.50],
    },
}
D = 2.0 * IN_TO_M 

def magnet_reluctance(l_m: float, mu_r: float, A_m:float ,) -> float : 
    """reluctance of the permanant magnet
    R_m = l_m /(mu * A_m)"""
    R_m = l_m / (MU_0 * mu_r * A_m)
    return R_m

def effective_gap_area(x,w,d): 
    """
    roters fringing correction for effective air gap area [m^2]
    A_eff = (w +2x)(d + 2x)
    because fringing flux bulges around the pole face edges it makes the effective gap area larger than the physical pole face w*d"""
    return (w + 2 * x) * (d + 2 * x)

def airgap_reluctance(x, A_eff): 
    """reluctance from both air gaps in series
    R_g = 2x / (MU_0 * A_eff"""
    R_g = (2 * x)/(MU_0 * A_eff) 
    return R_g 

def flux_calculation(MMF, R_m, R_g): 
    """flux flux circuit [Wb]
    phi = MMF / (R_tot)"""
    R_tot = R_m + R_g 
    flux = MMF / R_tot 
    return flux 

def adhesion_force(flux, A_eff):
    """adhesion force [N]
    Force = phi^2 / (A_eff * MU_0)  """
    adhesion = flux**2 / (A_eff * MU_0)
    return adhesion 

def run_lpr(magnet_params, w_in):
    """
    Run the full LPR force sweep for one magnet + pole width config.
 
    Args:
        magnet_params : dict — one entry from the MAGNETS dict above
        w_in          : float — pole width in inches
 
    Returns:
        dict with arrays: x [m], F [N], kappa [-], A_eff [m²]
        and scalars:      w [m], w_in [in], MMF [A], R_m [A/Wb]
    """
    Br   = magnet_params["Br"]
    Hc   = magnet_params["Hc"]
    l_m  = magnet_params["l_m"]
    A_m  = magnet_params["A_m"]
    mu_r = Br / (MU_0 * Hc)
    MMF = Hc * l_m                            
    R_m = magnet_reluctance(l_m, mu_r, A_m)  
    w   = w_in * IN_TO_M     
    F_arr     = np.zeros(N_PTS)
    A_eff_arr = np.zeros(N_PTS)
    for i, x in enumerate(X_ARR):
        A_eff        = effective_gap_area(x, w, D)
        R_g = airgap_reluctance(x, A_eff)
        Phi = flux_calculation(MMF, R_m, R_g)

        F_arr[i]     = adhesion_force(Phi, A_eff)
        A_eff_arr[i] = A_eff
    return {
        "x"      : X_ARR.copy(),   
        "F"      : F_arr,          
        "kappa"  : X_ARR / w,     
        "A_eff"  : A_eff_arr,      
        "w"      : w,              
        "w_in"   : w_in,           
        "MMF"    : MMF,            
        "R_m"    : R_m,           
        "mu_r"   : mu_r,    
        }