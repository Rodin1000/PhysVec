import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_Jastrow(g, *, s=0.5, total_sz=None):
    # a verifier hilbert for spin0_5_Jastrow system: creates a spin-1/2 Hilbert space for Jastrow calculations
    # g: graph object representing the lattice structure (e.g., nk.graph.Square)
    # s: spin quantum number (default: 0.5)
    # total_sz: optional total spin projection constraint (default: None)
    #
    # Creates a Hilbert space for spin-1/2 particles using the computational basis
    # where configurations are represented as arrays of +1 (spin up) and -1 (spin down)
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi