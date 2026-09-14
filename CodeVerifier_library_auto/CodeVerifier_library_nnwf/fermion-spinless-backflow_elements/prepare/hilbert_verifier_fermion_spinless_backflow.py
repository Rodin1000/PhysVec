import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_fermion_spinless_backflow(N: int, N_f: int):
    # a verifier hilbert for fermion_spinless_backflow system: creates a Hilbert space for spinless fermions with fixed particle number
    # N: number of sites/orbitals in the lattice
    # N_f: number of fermions in the system (particle number)
    #
    # Creates a Hilbert space for spinless fermions using SpinOrbitalFermions with s=None
    # and fixed particle number conservation via n_fermions parameter
    #
    hi = nk.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)
    return hi