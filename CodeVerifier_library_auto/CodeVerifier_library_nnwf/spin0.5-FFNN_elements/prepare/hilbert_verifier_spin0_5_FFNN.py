import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_FFNN(g, *, s: float = 0.5):
    # a verifier hilbert for spin0_5_FFNN system: creates a Hilbert space for spin-1/2 system with given graph
    # g: graph object representing the lattice structure, must have n_nodes attribute
    # s: spin value, default is 0.5 for spin-1/2 systems
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes)
    #
    return hi