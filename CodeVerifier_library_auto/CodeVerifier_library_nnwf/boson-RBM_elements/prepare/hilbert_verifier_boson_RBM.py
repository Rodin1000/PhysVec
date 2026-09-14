import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_boson_RBM(Lx: int, Ly: int, n_max: int, *, pbc: bool = True, n_particles: int = None):
    # a verifier hilbert for boson_RBM system: creates a 2D square lattice graph and Fock Hilbert space for bosons
    # Lx: number of sites along x-direction
    # Ly: number of sites along y-direction
    # n_max: maximum occupation number per site (inclusive)
    # pbc: periodic boundary conditions (default True)
    # n_particles: constraint on total number of particles (default None, no constraint)
    
    # Create the 2D square lattice graph
    g = nk.graph.Square(extent=[Lx, Ly], pbc=pbc)
    
    # Create the Fock Hilbert space for bosons
    hi = nk.hilbert.Fock(n_max=n_max, N=g.n_nodes, n_particles=n_particles)
    
    return g, hi
