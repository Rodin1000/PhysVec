from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_statevector(hamiltonian, circinit: QuantumCircuit, circevol: QuantumCircuit):
    # a verifier effector for spin0_5_statevector system: combines initial state and evolution circuits, then calculates the final statevector
    # hamiltonian: the Hamiltonian operator for the system (used for context, not directly in statevector calculation)
    # circinit: QuantumCircuit representing the initial state preparation
    # circevol: QuantumCircuit representing the time evolution operation
    
    # Combine the initial state circuit and evolution circuit
    # Create a new circuit with the same number of qubits as the initial circuit
    num_qubits = circinit.num_qubits
    combined_circuit = QuantumCircuit(num_qubits)
    
    # Append the initial state circuit
    combined_circuit.compose(circinit, inplace=True)
    
    # Append the evolution circuit
    combined_circuit.compose(circevol, inplace=True)
    
    # Calculate the final statevector using Statevector.from_instruction
    final_statevector = Statevector.from_instruction(combined_circuit)
    
    return final_statevector