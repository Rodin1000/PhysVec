import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin):
    # a verifier observable for spin0_5_tVMC system: define a total magnetization observable
    # hi: the spin-1/2 Hilbert space object
    #
    # The observable is defined as the sum of sigma_z operators over all sites
    n_sites = hi.size
    obs = sum([nk.operator.spin.sigmaz(hi, i) for i in range(n_sites)])
    
    return obs
