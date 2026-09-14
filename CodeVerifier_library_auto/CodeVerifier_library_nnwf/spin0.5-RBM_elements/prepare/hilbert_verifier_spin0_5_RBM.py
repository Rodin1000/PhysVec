import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_RBM(g, *, s: float = 0.5, total_sz: float = None):
    # a verifier hilbert for spin0_5_RBM system: creates a spin-1/2 Hilbert space for RBM calculations
    # g: graph object defining the lattice structure and number of sites
    # s: spin value, set to 0.5 for spin-1/2 systems (default: 0.5)
    # total_sz: optional constraint for total magnetization (sum of spins), if None no constraint is applied
    #
    # Create spin-1/2 Hilbert space with N sites from graph
    if total_sz is not None:
        # Apply sum constraint for fixed total magnetization
        constraint = nk.constraints.SumConstraint(total_sz=total_sz)
        hi = nk.hilbert.Spin(s=s, N=g.n_nodes, constraint=constraint)
    else:
        # No constraint on total magnetization
        hi = nk.hilbert.Spin(s=s, N=g.n_nodes)
    #
    return hi