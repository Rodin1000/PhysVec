import netket as nk
import numpy as np

# observable-----------------------------------------------------------------------------------------
def observable_verifier_boson_RBM(hi, site: int):
    # a verifier observable for boson_RBM system: defines a local number operator observable on a specific site
    # hi: Hilbert space object (e.g., nk.hilbert.Fock) for the bosonic system
    # site: site index where the number operator is measured
    
    # Create the number operator observable on the specified site
    # For bosonic systems, we use the number operator n = b†b
    obs = nk.operator.boson.number(hi, site)
    
    return obs
