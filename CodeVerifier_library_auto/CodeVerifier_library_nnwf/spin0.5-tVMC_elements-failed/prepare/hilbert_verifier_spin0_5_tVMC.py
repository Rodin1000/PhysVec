import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_tVMC(g, *, s=1/2, total_sz=None):
    # a verifier hilbert for spin0_5_tVMC system: creates a spin-1/2 Hilbert space for a given graph
    # g: graph object representing the lattice structure (e.g., nk.graph.Square)
    # s: spin quantum number (default: 1/2)
    # total_sz: optional constraint on total magnetization (default: None)
    #
    # Creates a Spin Hilbert space object with spin quantum number s=1/2
    # and number of sites equal to the number of nodes in graph g
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi