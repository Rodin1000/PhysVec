import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli
import qiskit_nature
import qiskit_algorithms
import qiskit_aer
from typing import Optional, Union

# circevol-----------------------------------------------------------------------------------------
def circevol_verifier_fermion_IQPE(
    hamiltonian: Union[SparsePauliOp, Pauli],
    base_time: float,
    eval_qubits: int,
    evolution: str = "Trotter",
    num_system_qubits: Optional[int] = None,
    append_barriers: bool = False,
) -> QuantumCircuit:
    # a verifier circevol for fermion_IQPE system: build controlled-U powers for IQPE without init/measure
    # hamiltonian: qubit operator for evolution (SparsePauliOp or Pauli), provided externally
    # base_time: base evolution time t0 used to construct powers U^{2^k}
    # eval_qubits: number of control/evaluation qubits for IQPE power sequence
    # evolution: synthesis method for PauliEvolutionGate (e.g., "Trotter", "Suzuki")
    # num_system_qubits: optional explicit number of system qubits; inferred from operator if None
    # append_barriers: whether to insert barriers between controlled powers (default False)

    # infer system size from operator if not provided
    sys_qubits = num_system_qubits if num_system_qubits is not None else getattr(hamiltonian, "num_qubits", None)
    if sys_qubits is None:
        raise ValueError("num_system_qubits must be provided when hamiltonian does not expose num_qubits")

    total_qubits = eval_qubits + sys_qubits
    qc = QuantumCircuit(total_qubits)

    control_indices = list(range(eval_qubits))
    system_indices = list(range(eval_qubits, total_qubits))

    # append controlled U^{2^k} for k in [0..eval_qubits-1]
    for k in range(eval_qubits):
        time_k = base_time * (2 ** k)
        evo_gate = PauliEvolutionGate(hamiltonian, time=time_k, synthesis=evolution)
        ctrl_evo = evo_gate.control(1)
        qc.append(ctrl_evo, [control_indices[k]] + system_indices)
        if append_barriers:
            qc.barrier()

    return qc
