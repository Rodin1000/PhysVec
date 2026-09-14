import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_FFNN(hi, g):
    # a verifier hamiltonian for spin0_5_FFNN system: creates a Heisenberg Hamiltonian for spin-1/2 system
    # hi: Hilbert space object for spin-1/2 degrees of freedom
    # g: graph object defining lattice structure and connectivity
    #
    # Creates a Heisenberg Hamiltonian using NetKet's built-in operator
    #
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
    return ha