"""

Author: Ken Aoki. 6/26/26
Model to predict the adhesion force of the magnets using the lumped parameter reluctance model 
This is a simple model that uses the geometry of the magnets and the material properties to predict the adhesion force. The model is based on the 
assumption that the magnets are in a parallel configuration and that the magnetic field is uniform across the surface of the magnets. 
The model also assumes that the magnets are in a vacuum and that there are no other magnetic materials present. 
The model simplifes the magnetic system into its equivealnt magnetic cicuit and uses the reluctance of the magnetic circuit to calculate the adhesion force."""

def internal_reluctance ()
    """will calculate the internal reluctance of the magnetic circuit 
    args: 
    l_m : magnet length in magnetization direction [m]
    A_m : magnet cross sectional area [m^2]
