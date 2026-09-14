from qiskit import QuantumCircuit
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock

# circinit-----------------------------------------------------------------------------------------
def circinit_verifier_spin0_5_statevector(num_spatial_orbitals: int, num_particles: tuple, qubit_mapper=None):
    # a verifier circinit for spin0_5_statevector system: creates a quantum circuit to initialize a spin-1/2 state using Hartree-Fock method
    # num_spatial_orbitals: number of spatial orbitals in the system
    # num_particles: tuple containing number of alpha and beta electrons (e.g., (1,1) for one of each)
    # qubit_mapper: qubit mapper object (default: JordanWignerMapper) that maps fermionic operators to qubits
    
    if qubit_mapper is None:
        qubit_mapper = JordanWignerMapper()
    
    # Create Hartree-Fock initial state circuit
    initial_state_circuit = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=qubit_mapper
    )
    
    return initial_state_circuit