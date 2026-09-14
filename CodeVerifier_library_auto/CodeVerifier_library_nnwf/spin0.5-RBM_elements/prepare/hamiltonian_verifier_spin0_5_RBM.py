import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_RBM(hi, g, *, J=-1.0, h=1.0):
    # a verifier hamiltonian for spin0_5_RBM system: creates a Heisenberg or Ising Hamiltonian for spin-1/2 RBM models
    # hi: Hilbert space object for spin-1/2 system (nk.hilbert.Spin)
    # g: graph/lattice object defining the system geometry (nk.graph)
    # J: coupling parameter for spin interactions (default: -1.0)
    # h: transverse field parameter (default: 1.0)
    
    # Create Ising Hamiltonian with given parameters
    ha = nk.operator.Ising(hilbert=hi, graph=g, J=J, h=h)
    
    return ha