from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import ParityMapper, JordanWignerMapper
from qiskit_nature.second_q.operators import SpinOp

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_statevector(num_qubits: int, interaction_terms: list = None, pbc: bool = True):
    # a verifier hamiltonian for spin0_5_statevector system: defines a spin-1/2 Hamiltonian using Pauli operators
    # num_qubits: number of qubits in the system (must be provided as input)
    # interaction_terms: list of tuples containing Pauli operator strings and coefficients, e.g. [("XX", 1.0), ("ZZ", -1.0)]
    # pbc: whether to use periodic boundary conditions (default: True)
    
    if interaction_terms is None:
        # Default to Heisenberg model with nearest-neighbor interactions
        interaction_terms = []
        for i in range(num_qubits):
            next_i = (i + 1) % num_qubits if pbc else i + 1
            if next_i < num_qubits:  # Avoid out-of-bounds for open boundary
                interaction_terms.extend([
                    ("XX", 1.0, i, next_i),
                    ("YY", 1.0, i, next_i),
                    ("ZZ", 1.0, i, next_i)
                ])
    
    # Create Pauli terms from interaction_terms
    pauli_list = []
    for term in interaction_terms:
        if len(term) == 2:  # Simple case: ("XX", 1.0)
            pauli_str, coeff = term
            # Apply to first two qubits as example
            pauli_list.append((pauli_str, coeff))
        elif len(term) == 4:  # Specific qubit indices: ("XX", 1.0, i, j)
            pauli_str, coeff, qubit_i, qubit_j = term
            # Create Pauli string for specific qubits
            full_pauli = ['I'] * num_qubits
            full_pauli[qubit_i] = pauli_str[0]
            full_pauli[qubit_j] = pauli_str[1]
            pauli_list.append((''.join(full_pauli), coeff))
    
    # Construct Hamiltonian using SparsePauliOp
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    
    return hamiltonian