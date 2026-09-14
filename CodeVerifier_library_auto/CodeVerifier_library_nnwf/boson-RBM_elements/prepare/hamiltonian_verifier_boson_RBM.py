import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_boson_RBM(hi, g, *, U: float = 1.0, V: float = 0.0, J: float = 1.0, mu: float = 0.0):
    # a verifier hamiltonian for boson_RBM system: Bose-Hubbard Hamiltonian for bosonic systems
    # hi: Fock Hilbert space object for bosons (e.g., nk.hilbert.Fock)
    # g: graph/lattice object defining the connectivity (e.g., nk.graph.Hypercube or nk.graph.Square)
    # U: on-site interaction strength (default 1.0)
    # V: density-density interaction strength between neighbors (default 0.0)
    # J: hopping amplitude (default 1.0)
    # mu: chemical potential (default 0.0)
    
    ha = nk.operator.BoseHubbard(hilbert=hi, U=U, V=V, J=J, mu=mu, graph=g)
    
    return ha
