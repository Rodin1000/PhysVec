from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import PauliSumOp

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_statevector(num_qubits: int):
    # a verifier observable for spin0_5_statevector system: defines observables as SparsePauliOp or PauliSumOp for measurement
    # num_qubits: number of qubits in the system
    
    # Create observables for spin-1/2 system
    # Example: Magnetization observable (sum of Z operators)
    magnetization_pauli_list = [("Z" + "I" * (num_qubits-1), 1.0)]
    for i in range(1, num_qubits):
        pauli_str = "I" * i + "Z" + "I" * (num_qubits-i-1)
        magnetization_pauli_list.append((pauli_str, 1.0))
    
    magnetization_op = SparsePauliOp.from_list(magnetization_pauli_list)
    
    # Return the observable
    return magnetization_op