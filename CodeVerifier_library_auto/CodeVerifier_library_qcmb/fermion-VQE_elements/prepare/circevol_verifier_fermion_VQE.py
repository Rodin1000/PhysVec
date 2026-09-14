from typing import Optional, Tuple, Union

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.opflow import PauliSumOp
from qiskit.circuit.library import PauliEvolutionGate

from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.mappers import QubitMapper, JordanWignerMapper

from qiskit_algorithms import VQE  # explicit import as required
from qiskit_aer import Aer  # explicit import as required


# circevol-----------------------------------------------------------------------------------------
def circevol_verifier_fermion_VQE(
    hamiltonian: Union[SparsePauliOp, PauliSumOp],
    num_spatial_orbitals: int,
    num_particles: Tuple[int, int],
    num_qubits: int,
    evolution_time: float,
    ansatz: Optional[QuantumCircuit] = None,
    qubit_mapper: Optional[QubitMapper] = None,
    reps: int = 1,
    insert_barrier: bool = True,
) -> QuantumCircuit:
    # a verifier circevol for fermion_VQE system: build a variational evolution circuit (UCCSD + Pauli evolution) without initialization or measurements
    # hamiltonian: qubit-space Hamiltonian operator for evolution (SparsePauliOp or PauliSumOp)
    # num_spatial_orbitals: number of spatial orbitals used to parameterize the UCCSD ansatz
    # num_particles: tuple of (n_alpha, n_beta) electrons for the electronic structure problem
    # num_qubits: total number of qubits; must be consistent with the mapped Hamiltonian and ansatz
    # evolution_time: total evolution time t for exp(-i H t)
    # ansatz: optional pre-built variational ansatz QuantumCircuit; if None, a UCCSD ansatz is constructed
    # qubit_mapper: mapper from qiskit-nature (e.g., JordanWignerMapper) used when constructing UCCSD
    # reps: number of first-order Trotter steps for splitting the evolution gate
    # insert_barrier: whether to insert a circuit barrier between ansatz and evolution (default True)

    # ensure we have a mapper if ansatz is to be constructed
    if ansatz is None:
        mapper = qubit_mapper if qubit_mapper is not None else JordanWignerMapper()
        # Build a UCCSD ansatz without explicit state initialization; VQE uses zeros initial point for HF
        ansatz = UCCSD(
            qubit_mapper=mapper,
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
        )

    # construct the circuit with the specified qubit count
    circuit = QuantumCircuit(num_qubits)

    # compose the parameterized ansatz (no measurements or initialize instructions)
    circuit.compose(ansatz, qubits=range(num_qubits), inplace=True)

    if insert_barrier:
        circuit.barrier()

    # build Trotterized evolution using PauliEvolutionGate; avoid measurement/initialization
    step_time = float(evolution_time) / float(max(1, reps))
    evo_gate = PauliEvolutionGate(hamiltonian, time=step_time)
    for _ in range(max(1, reps)):
        circuit.append(evo_gate, circuit.qubits)

    return circuit
