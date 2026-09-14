import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_Jastrow(hi, g):
    # a verifier hamiltonian for spin0_5_Jastrow system: creates a Heisenberg Hamiltonian for spin-1/2 system with zero magnetization
    # hi: Hilbert space object for spin-1/2 degrees of freedom with zero magnetization constraint
    # g: graph object representing the lattice structure
    #
    # Creates a Heisenberg Hamiltonian using the provided Hilbert space and graph
    # The Hamiltonian captures short-range spin correlations as in the Jastrow model
    #
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
    return ha
