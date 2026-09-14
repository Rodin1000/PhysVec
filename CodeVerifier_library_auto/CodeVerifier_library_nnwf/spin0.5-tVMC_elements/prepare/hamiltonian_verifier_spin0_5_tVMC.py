import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin, g: nk.graph.Graph, J: float, h: float):
    # a verifier hamiltonian for spin0_5_tVMC system: transverse field Ising model on a graph
    # hi: the Hilbert space object for spin-1/2 particles
    # g: the graph or lattice object defining the connectivity
    # J: the interaction strength between adjacent spins
    # h: the transverse field strength
    #
    ha = nk.operator.Ising(hilbert=hi, graph=g, J=J, h=h)
    #
    return ha
