import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_tVMC(hi, g, *, J=1.0, h=1.0):
    # a verifier hamiltonian for spin0_5_tVMC system: transverse field Ising model with tunable parameters
    # hi: Hilbert space object for spin-1/2 system (nk.hilbert.Spin)
    # g: graph object defining lattice structure (nk.graph)
    # J: coupling strength for spin-spin interactions (default: 1.0)
    # h: strength of transverse magnetic field (default: 1.0)
    
    # Construct transverse field Ising model Hamiltonian
    # Using LocalOperator to build custom Hamiltonian
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h)
    
    return ha