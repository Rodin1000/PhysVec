import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_boson_FFNN(hi, g, *, U: float = 1.0, V: float = 0.0, J: float = 1.0, mu: float = 0.0):
    # a verifier hamiltonian for boson_FFNN system: Bose-Hubbard Hamiltonian with on-site and density-density interactions
    # hi: Fock Hilbert space object for bosons
    # g: graph/lattice object defining the connectivity
    # U: on-site interaction strength (default 1.0)
    # V: density-density interaction strength between neighboring sites (default 0.0)
    # J: hopping amplitude between neighboring sites (default 1.0)
    # mu: chemical potential (default 0.0)
    
    ha = nk.operator.BoseHubbard(hilbert=hi, U=U, V=V, J=J, mu=mu, graph=g)
    
    return ha
