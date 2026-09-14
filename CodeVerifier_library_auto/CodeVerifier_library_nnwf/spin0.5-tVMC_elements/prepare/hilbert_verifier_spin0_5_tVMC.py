import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_tVMC(g, *, s: float = 0.5, total_sz: float = None):
    # a verifier hilbert for spin0_5_tVMC system: defines a spin-1/2 Hilbert space on a given graph
    # g: the graph or lattice object defining the number of sites
    # s: the spin quantum number (default 0.5 for spin-1/2)
    # total_sz: optional total magnetization constraint (sum of Sz)
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi
