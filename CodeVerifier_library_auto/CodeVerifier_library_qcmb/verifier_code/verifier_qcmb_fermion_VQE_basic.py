from __future__ import annotations

import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import qiskit
import qiskit_algorithms  # keep explicit base-module import
import qiskit_aer  # noqa: F401
import qiskit_nature  # noqa: F401

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, PauliList
# Compatibility: PauliSumOp moved/removed in newer Qiskit versions
try:  # Qiskit < 1.0
    from qiskit.opflow import PauliSumOp  # type: ignore
except Exception:  # Qiskit >= 1.0
    PauliSumOp = None  # type: ignore
from qiskit.circuit.library import PauliEvolutionGate
# Compatibility: Estimator primitive location varies across Qiskit versions
try:
    from qiskit.primitives import Estimator  # Terra >= 0.22
except Exception:
    try:
        from qiskit_aer.primitives import Estimator  # Aer fallback
    except Exception:
        Estimator = None  # type: ignore

from qiskit_aer import Aer
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer, COBYLA

from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit_nature.second_q.mappers import (
    QubitMapper,
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.operators import FermionicOp


# === circevol_verifier_fermion_VQE.py ===

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


# === circinit_verifier_fermion_VQE.py ===

def circinit_verifier_fermion_VQE(
    num_qubits: int,
    occupation: Optional[Sequence[int]] = None,
    num_spatial_orbitals: Optional[int] = None,
    num_particles: Optional[Tuple[int, int]] = None,
    qubit_mapper: Optional[QubitMapper] = None,
) -> QuantumCircuit:
    # a verifier circinit for fermion_VQE system: prepare initial state circuit via occupation bitstring or Hartree–Fock
    # num_qubits: total number of qubits for the initialization circuit
    # occupation: bitstring-like sequence (length num_qubits) marking occupied qubits to flip with X
    # num_spatial_orbitals: number of spatial orbitals for Hartree–Fock initial state (HF path)
    # num_particles: (n_alpha, n_beta) electrons tuple for HF initial state (HF path)
    # qubit_mapper: qiskit-nature QubitMapper to build HF initial state on num_qubits (HF path)

    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer")

    # Path 1: explicit occupation bitstring -> apply X on occupied positions
    if occupation is not None:
        if len(occupation) != num_qubits:
            raise ValueError("occupation length must equal num_qubits")
        qc = QuantumCircuit(num_qubits)
        for i, occ in enumerate(occupation):
            if bool(occ):
                qc.x(i)
        return qc

    # Path 2: Hartree–Fock circuit from qiskit-nature (no measurement/evolution)
    if (num_spatial_orbitals is not None) and (num_particles is not None) and (qubit_mapper is not None):
        hf_circ = HartreeFock(num_spatial_orbitals, num_particles, qubit_mapper)
        if hf_circ.num_qubits != num_qubits:
            raise ValueError(
                f"Hartree–Fock circuit qubit count ({hf_circ.num_qubits}) does not match num_qubits ({num_qubits})"
            )
        return hf_circ

    # Default: return the |0...0> state preparation (empty circuit)
    return QuantumCircuit(num_qubits)


# === effector_verifier_fermion_VQE.py ===

def effector_verifier_fermion_VQE(
    hamiltonian: SparsePauliOp,
    circinit: QuantumCircuit | None,
    circevol: QuantumCircuit,
    estimator: Estimator,
    optimizer: Optimizer,
    initial_point: np.ndarray | None = None,
    prepend_init: bool = True,
) -> dict:
    # a verifier effector for fermion_VQE system: combine initial/evolution circuits and run VQE to obtain energy and statevector
    # hamiltonian: qubit operator to evaluate with VQE (e.g., SparsePauliOp mapped from fermionic Hamiltonian)
    # circinit: initial state preparation circuit to prepend before the ansatz (can be None if not used)
    # circevol: parameterized evolution/ansatz circuit used by VQE
    # estimator: Qiskit Estimator primitive instance used by VQE for expectation evaluations
    # optimizer: classical optimizer instance (from qiskit_algorithms.optimizers) for parameter updates
    # initial_point: optional array of initial parameter values for the ansatz
    # prepend_init: whether to prepend circinit before circevol when circinit is provided

    # Prepare the ansatz by optionally composing the initial-state circuit before the evolution circuit
    if circinit is not None and prepend_init:
        ansatz = circinit.compose(circevol, front=False)
    else:
        ansatz = circevol

    # Configure and run VQE
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

    # Extract optimized circuit: prefer provided optimal_circuit; otherwise bind optimal_point
    optimal_circuit = getattr(result, "optimal_circuit", None)
    if optimal_circuit is None and getattr(result, "optimal_point", None) is not None:
        try:
            optimal_circuit = ansatz.bind_parameters(result.optimal_point)
        except Exception:
            # Fallback for older Qiskit versions
            optimal_circuit = ansatz.assign_parameters(result.optimal_point, inplace=False)

    # Compute final statevector from the optimized circuit when available
    statevector = None
    if optimal_circuit is not None:
        try:
            # Bind any remaining free parameters to numeric values (default 0.0) before simulation
            if getattr(optimal_circuit, "parameters", None):
                param_binds = {p: 0.0 for p in optimal_circuit.parameters}
                try:
                    tmp_circ = optimal_circuit.assign_parameters(param_binds, inplace=False)
                except Exception:
                    tmp_circ = optimal_circuit.bind_parameters(param_binds)
            else:
                tmp_circ = optimal_circuit
            statevector = Statevector.from_instruction(tmp_circ)
        except Exception:
            # If parameter binding or simulation fails, skip statevector generation
            statevector = None

    # Prepare outputs: energy (real), eigenvalue (possibly complex), and statevector
    energy = None
    eigenvalue = getattr(result, "eigenvalue", None)
    if eigenvalue is not None:
        try:
            energy = float(np.real(eigenvalue))
        except Exception:
            energy = None

    return {
        "energy": energy,
        "eigenvalue": eigenvalue,
        "statevector": statevector,
        "optimal_point": getattr(result, "optimal_point", None),
        "optimal_parameters": getattr(result, "optimal_parameters", None),
        "optimal_circuit": optimal_circuit,
        "vqe_result": result,
    }


# === hamiltonian_verifier_fermion_VQE.py ===

def hamiltonian_verifier_fermion_VQE(
    terms: Dict[str, complex],
    num_spin_orbitals: int,
    mapper: str = "jordan_wigner",
    num_particles: Optional[Tuple[int, int]] = None,
    make_hermitian: bool = True,
    simplify: bool = True,
) -> SparsePauliOp:
    # a verifier hamiltonian for fermion_VQE system: build fermionic Hamiltonian and map to qubit operator
    # terms: dictionary of FermionicOp terms {"op_string": coefficient}
    # num_spin_orbitals: total number of spin orbitals (register length for FermionicOp)
    # mapper: fermion-to-qubit mapping name ("jordan_wigner", "parity", or "bravyi_kitaev")
    # num_particles: tuple (n_alpha, n_beta); enables two-qubit reduction for parity mapping
    # make_hermitian: if True, symmetrize the FermionicOp as (H + H^†)/2 before mapping
    # simplify: if True, call FermionicOp.simplify() to combine/cancel terms before mapping

    # Construct the second-quantized Fermionic Hamiltonian
    ferm_op = FermionicOp(terms, num_spin_orbitals=num_spin_orbitals)

    if simplify:
        ferm_op = ferm_op.simplify()

    if make_hermitian:
        # Ensure Hermiticity: (H + H^†)/2
        ferm_op = (ferm_op + ferm_op.adjoint()) * 0.5

    # Select mapper
    name = (mapper or "").strip().lower()
    if name in {"jordan_wigner", "jordanwigner", "jw", "jordan-wigner"}:
        mapper_obj = JordanWignerMapper()
    elif name in {"parity", "paritymapper"}:
        # num_particles enables two-qubit reduction when provided
        mapper_obj = ParityMapper(num_particles=num_particles) if num_particles is not None else ParityMapper()
    elif name in {"bravyi_kitaev", "bravyikitaev", "bk", "bravyi-kitaev"}:
        mapper_obj = BravyiKitaevMapper()
    else:
        raise ValueError(f"Unsupported mapper '{mapper}'. Use 'jordan_wigner', 'parity', or 'bravyi_kitaev'.")

    # Map the Fermionic Hamiltonian to a qubit-space SparsePauliOp
    qubit_h = mapper_obj.map(ferm_op)

    return qubit_h


# === observable_verifier_fermion_VQE.py ===

def observable_verifier_fermion_VQE(
    num_qubits: int,
    observables_pauli: Optional[Dict[str, List[Tuple[str, complex]]]] = None,
    second_q_observables: Optional[Dict[str, FermionicOp]] = None,
    mapper: Optional[QubitMapper] = None,
    output_type: str = "SparsePauliOp",
    simplify_op: bool = True,
    group: bool = False,
    qubit_wise: bool = False,
) -> Dict[str, Union[SparsePauliOp, PauliSumOp, List[Union[SparsePauliOp, PauliSumOp]]]]:
    # a verifier observable for fermion_VQE system: constructs measurement observables as SparsePauliOp/PauliSumOp with optional commuting-group partitioning
    # num_qubits: total number of qubits for the target register; all constructed observables must match this size
    # observables_pauli: dict mapping observable names to lists of (pauli_label, coeff) tuples used to build SparsePauliOp
    # second_q_observables: dict mapping observable names to qiskit-nature FermionicOp objects to be mapped to qubits
    # mapper: qiskit-nature QubitMapper to map second-quantized operators to qubit operators (required if second_q_observables is provided)
    # output_type: one of {"SparsePauliOp", "PauliSumOp"}; selects the desired output operator type
    # simplify_op: whether to simplify each operator (combine duplicates/remove zeros) before optional grouping
    # group: whether to partition each observable into commuting groups for measurement optimization
    # qubit_wise: if True, perform qubit-wise commutation grouping; otherwise, full commutation grouping

    if observables_pauli is None and second_q_observables is None:
        raise ValueError("At least one of 'observables_pauli' or 'second_q_observables' must be provided.")

    if second_q_observables is not None and mapper is None:
        raise ValueError("'mapper' must be provided when 'second_q_observables' is specified.")

    prepared: Dict[str, Union[SparsePauliOp, PauliSumOp, List[Union[SparsePauliOp, PauliSumOp]]]] = {}

    # 1) Build from explicit Pauli term lists
    if observables_pauli is not None:
        for name, term_list in observables_pauli.items():
            # Construct SparsePauliOp from list of (label, coeff)
            op = SparsePauliOp.from_list(term_list)
            if op.num_qubits != num_qubits:
                raise ValueError(f"Observable '{name}' has {op.num_qubits} qubits but expected {num_qubits}.")
            if simplify_op:
                op = op.simplify()

            if group:
                groups = op.group_commuting(qubit_wise=qubit_wise)
                if output_type == "PauliSumOp":
                    if PauliSumOp is None:
                        raise ImportError("PauliSumOp is not available in this Qiskit version. Use output_type='SparsePauliOp' instead.")
                    prepared[name] = [PauliSumOp(g) for g in groups]
                else:
                    prepared[name] = groups
            else:
                if output_type == "PauliSumOp":
                    if PauliSumOp is None:
                        raise ImportError("PauliSumOp is not available in this Qiskit version. Use output_type='SparsePauliOp' instead.")
                    prepared[name] = PauliSumOp(op)
                else:
                    prepared[name] = op

    # 2) Map second-quantized observables via the provided mapper
    if second_q_observables is not None:
        for name, ferm_op in second_q_observables.items():
            qubit_op = mapper.map(ferm_op)
            if not isinstance(qubit_op, SparsePauliOp):
                # Defensive: ensure we end with a SparsePauliOp for measurement
                qubit_op = SparsePauliOp(qubit_op.primitive) if hasattr(qubit_op, "primitive") else SparsePauliOp(PauliList(["I" * num_qubits]), coeffs=[0.0])
            if qubit_op.num_qubits != num_qubits:
                raise ValueError(f"Mapped observable '{name}' has {qubit_op.num_qubits} qubits but expected {num_qubits}.")
            if simplify_op:
                qubit_op = qubit_op.simplify()

            if group:
                groups = qubit_op.group_commuting(qubit_wise=qubit_wise)
                if output_type == "PauliSumOp":
                    if PauliSumOp is None:
                        raise ImportError("PauliSumOp is not available in this Qiskit version. Use output_type='SparsePauliOp' instead.")
                    prepared[name] = [PauliSumOp(g) for g in groups]
                else:
                    prepared[name] = groups
            else:
                if output_type == "PauliSumOp":
                    if PauliSumOp is None:
                        raise ImportError("PauliSumOp is not available in this Qiskit version. Use output_type='SparsePauliOp' instead.")
                    prepared[name] = PauliSumOp(qubit_op)
                else:
                    prepared[name] = qubit_op

    return prepared


# === main integration ===

def main() -> dict:
    # Minimal parameters for quick verification
    num_spatial_orbitals = 2
    num_spin_orbitals = 2 * num_spatial_orbitals
    num_particles: Tuple[int, int] = (1, 1)

    # Simple fermionic number operator Hamiltonian (Hermitian)
    terms: Dict[str, complex] = {
        "+_0 -_0": 1.0,
        "+_1 -_1": 1.0,
        "+_2 -_2": 1.0,
        "+_3 -_3": 1.0,
    }

    # 1) Map Hamiltonian to qubits
    qubit_h = hamiltonian_verifier_fermion_VQE(
        terms=terms,
        num_spin_orbitals=num_spin_orbitals,
        mapper="jordan_wigner",
        num_particles=num_particles,
        make_hermitian=True,
        simplify=True,
    )
    num_qubits = qubit_h.num_qubits

    # 2) Build initial-state circuit (default |0...0>)
    circ_init = circinit_verifier_fermion_VQE(num_qubits=num_qubits)

    # 3) Build variational evolution circuit (UCCSD + short Pauli evolution)
    circevol = circevol_verifier_fermion_VQE(
        hamiltonian=qubit_h,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        num_qubits=num_qubits,
        evolution_time=0.1,
        ansatz=None,
        qubit_mapper=None,
        reps=1,
        insert_barrier=True,
    )

    # 4) Prepare a couple of simple observables for demonstration
    if num_qubits >= 2:
        obs_pauli = {
            "Z0": [("Z" + "I" * (num_qubits - 1), 1.0)],
            "Z1": [("I" + "Z" + "I" * (num_qubits - 2), 1.0)],
        }
    else:
        obs_pauli = {"Z0": [("Z", 1.0)]}

    _observables = observable_verifier_fermion_VQE(
        num_qubits=num_qubits,
        observables_pauli=obs_pauli,
        second_q_observables=None,
        mapper=None,
        output_type="SparsePauliOp",
        simplify_op=True,
        group=False,
        qubit_wise=False,
    )

    # 5) Run VQE using a modern Estimator (StatevectorEstimator or EstimatorV2) and a lightweight optimizer
    try:
        from qiskit.primitives import StatevectorEstimator  # Preferred when available
        estimator = StatevectorEstimator(seed=42)
    except Exception:
        try:
            # Prefer BackendEstimatorV2 if available (works with AerSimulator)
            from qiskit.primitives import BackendEstimatorV2
            from qiskit_aer import AerSimulator
            estimator = BackendEstimatorV2(backend=AerSimulator())
        except Exception:
            try:
                # Fallback to Aer EstimatorV2
                from qiskit_aer.primitives import EstimatorV2
                estimator = EstimatorV2()
            except Exception as exc:
                raise ImportError("No compatible EstimatorV2 or StatevectorEstimator available for VQE.") from exc
    optimizer = COBYLA(maxiter=5)

    result = effector_verifier_fermion_VQE(
        hamiltonian=qubit_h,
        circinit=circ_init,
        circevol=circevol,
        estimator=estimator,
        optimizer=optimizer,
        initial_point=None,
        prepend_init=True,
    )

    print("VQE energy:", result.get("energy"))
    if result.get("optimal_point") is not None:
        print("Optimal point length:", len(result["optimal_point"]))

    return result


if __name__ == "__main__":
    main()
