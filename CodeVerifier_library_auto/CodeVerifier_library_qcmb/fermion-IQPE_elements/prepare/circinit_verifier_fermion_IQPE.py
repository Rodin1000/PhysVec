from qiskit import QuantumCircuit
import qiskit_nature
from qiskit_nature.second_q.mappers import QubitMapper
import qiskit_algorithms
import qiskit_aer

# circinit-----------------------------------------------------------------------------------------
def circinit_verifier_fermion_IQPE(
    num_qubits: int,
    initial_bitstring: list[int] | None = None,
    amplitudes: list[complex] | None = None,
    qubit_order: list[int] | None = None,
    qubit_mapper: QubitMapper | None = None,
) -> QuantumCircuit:
    # a verifier circinit for fermion_IQPE system: prepares a fermionic initial state circuit without evolution or measurement
    # num_qubits: number of qubits in the mapped fermionic register
    # initial_bitstring: mapped occupation bitstring (length == num_qubits); apply X on qubits with value 1
    # amplitudes: optional statevector amplitudes of length 2**num_qubits to initialize the register
    # qubit_order: optional list of qubit indices specifying the target qubits for initialization (default is range(num_qubits))
    # qubit_mapper: optional Qiskit Nature qubit mapper that defines the fermion-to-qubit mapping (passed as dependency, not used for computation here)

    # select target qubits order
    q_indices = list(range(num_qubits)) if qubit_order is None else list(qubit_order)
    if len(q_indices) != num_qubits:
        raise ValueError("qubit_order length must equal num_qubits")

    # build a state-preparation-only circuit
    circuit = QuantumCircuit(num_qubits)

    # initialize via amplitudes if provided
    if amplitudes is not None:
        if len(amplitudes) != (1 << num_qubits):
            raise ValueError("amplitudes length must be 2**num_qubits")
        circuit.initialize(amplitudes, q_indices)
        return circuit

    # otherwise prepare via occupation bitstring if provided
    if initial_bitstring is not None:
        if len(initial_bitstring) != num_qubits:
            raise ValueError("initial_bitstring length must equal num_qubits")
        for i, bit in enumerate(initial_bitstring):
            if bit:
                circuit.x(q_indices[i])
        return circuit

    # default to |0...0> if no inputs provided
    return circuit
