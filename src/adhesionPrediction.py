"""

Author: Ken Aoki. 6/26/26
Model to predict the adhesion force of the magnets using the lumped parameter reluctance model 
This is a simple model that uses the geometry of the magnets and the material properties to predict the adhesion force. The model is based on the 
assumption that the magnets are in a parallel configuration and that the magnetic field is uniform across the surface of the magnets. 
The model also assumes that the magnets are in a vacuum and that there are no other magnetic materials present. 
The model simplifes the magnetic system into its equivealnt magnetic cicuit and uses the reluctance of the magnetic circuit to calculate the adhesion force."""

def internal_reluctance (magnet_length: float, mu_0: float, mu_r: float, Pole_Area: float) -> float:
    """will calculate the internal reluctance of the magnetic circuit 
    args: 
    l_m : magnet length in magnetization direction [m]
    A_m : magnet cross sectional area [m^2]
    mu: magnetic permeability of the magnet [H/m] 
    returns: 
    R_m: internal reluctance of the magnet [H^-1]"""
    R_m = magnet_length / (mu_r * mu_0 * Pole_Area)
    return R_m 
def airgap_reluctance(airgap: float, mu_0: float, A_m: float):
    """ will calcuate the reluctance due to the air gap
    args: 
    x: airgap of pole face and iron plate [m]
    mu_0 permeability of free space
    A_m cross sectional area [m^2]
    returns:
    R_g: reluctance due to air gap H/m"""
    R_g = airgap / (mu_0 * A_m)
    return R_g 
def flux_calculation(R_g: float, R_m: float, MMF: float) -> float: 
    """use ohms law equivalent to calculate the magnetic flux using MMF = flux * reluctance
    args: 
    MMF = magnetomotive force [H]
    R_g 
    R_m 
    return: 
    flux    """
    R_total = R_g + R_m 
    Magnetic_Flux = MMF / R_total 
    return Magnetic_Flux 

def Adhesion_Calculation(Magnetic_Flux: float, mu_0: float, Pole_Area: float) -> float: 
    "get force from flux mu and area"
    AdhesionForce = Magnetic_Flux**2 / (2 * mu_0 * Pole_Area)
    return AdhesionForce 
