import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_Jastrow(hi, *, indices=None):
    # a verifier observable for spin0_5_Jastrow system: defines observables for spin-1/2 Jastrow wave function calculations
    # hi: Hilbert space object for spin-1/2 system
    # indices: list of site indices to define local observables (default: all sites)
    #
    # Create magnetization observables as example observables
    if indices is None:
        indices = list(range(hi.size))
    
    # Example observable: total magnetization in x direction
    obs = nk.operator.spin.sigmax(hi, indices[0])
    for i in indices[1:]:
        obs += nk.operator.spin.sigmax(hi, i)
    
    return obs