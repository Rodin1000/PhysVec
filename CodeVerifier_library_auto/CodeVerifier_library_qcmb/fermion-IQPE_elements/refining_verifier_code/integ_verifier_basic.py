# Consolidated imports for fermion_IQPE verifier integration
import math
import qiskit
import qiskit_nature
import qiskit_algorithms
import qiskit_aer

from typing import Optional, Union, Dict, Any, List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli, Statevector
from qiskit_algorithms.phase_estimators import IterativePhaseEstimation
from qiskit_aer.primitives import Sampler
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevSuperFastMapper,
    QubitMapper,
)


# === circevol_verifier_fermion_IQPE.py ===
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


# === circinit_verifier_fermion_IQPE.py ===
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


# === effector_verifier_fermion_IQPE.py ===
# effector-----------------------------------------------------------------------------------------
def effector_verifier_fermion_IQPE(
    hamiltonian: SparsePauliOp,
    circinit: QuantumCircuit,
    circevol: QuantumCircuit,
    time: float = 1.0,
    num_iterations: int = 10,
    sampler: Optional[Sampler] = None,
    identity_energy_shift: float = 0.0,
) -> Dict[str, Any]:
    # a verifier effector for fermion_IQPE system: run iterative phase estimation on provided evolution with prepared fermionic state
    # hamiltonian: problem Hamiltonian as SparsePauliOp used to construct the evolution (for reference and validation)
    # circinit: initial state preparation circuit (candidate eigenstate of H)
    # circevol: evolution circuit representing exp(-i * time * H)
    # time: evolution time used in U = exp(-i * time * H) for energy conversion (default value)
    # num_iterations: number of IQPE iterations controlling phase precision (default value)
    # sampler: sampler primitive used by IterativePhaseEstimation for execution (must be provided)
    # identity_energy_shift: optional constant shift to add to energy if identity terms were removed (default value)

    if sampler is None:
        raise ValueError("sampler must be provided for IterativePhaseEstimation execution")

    # Prepare combined circuit for statevector inspection: apply evolution after state preparation
    combined = circinit.compose(circevol)

    # Run IQPE using provided unitary (evolution) and state preparation
    ipe = IterativePhaseEstimation(num_iterations=num_iterations)
    result = ipe.estimate(unitary=circevol, state_preparation=circinit, sampler=sampler)

    # Extract phase and convert to energy using E ≈ (2π * phase) / time plus optional shift
    phase = float(result.phase)
    angle = 2.0 * math.pi * phase
    energy = angle / float(time) + float(identity_energy_shift)

    # Build statevectors for verification/reporting
    sv_init = Statevector.from_instruction(circinit)
    sv_evolved = Statevector.from_instruction(combined)

    return {
        "phase": phase,
        "angle": angle,
        "energy": energy,
        "num_iterations": int(result.num_iterations),
        "statevector_init": sv_init,
        "statevector_evolved": sv_evolved,
    }


# === hamiltonian_verifier_fermion_IQPE.py ===
# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_fermion_IQPE(
    num_qubits: int,
    num_spin_orbitals: int,
    one_body_terms: list,
    two_body_terms: list | None = None,
    mapper: str = "jordan_wigner",
    num_particles: tuple[int, int] | None = None,
    two_qubit_reduction: bool = False,
) -> SparsePauliOp:
    # a verifier hamiltonian for fermion_IQPE system: build a fermionic second-quantized Hamiltonian and map it to qubits
    # num_qubits: target number of qubits after mapping (used for validation)
    # num_spin_orbitals: total number of spin orbitals in the fermionic system
    # one_body_terms: list of (p, q, coeff) for a_p^† a_q; real coeffs recommended; hermitian conjugates are added
    # two_body_terms: optional list of (p, q, r, s, coeff) for a_p^† a_q^† a_r a_s; hermitian conjugates are added (default None)
    # mapper: qubit mapping to use: 'jordan_wigner', 'parity', or 'bksf' (default 'jordan_wigner')
    # num_particles: (n_alpha, n_beta) used for parity two-qubit reduction when applicable (default None)
    # two_qubit_reduction: enable two-qubit reduction with parity mapping if supported (default False)

    # assemble fermionic operator terms ensuring hermiticity by adding conjugate counterparts
    term_dict: dict[str, complex] = {}

    def add_term(label: str, coeff: float) -> None:
        term_dict[label] = term_dict.get(label, 0.0) + float(coeff)

    # validate indices and accumulate one-body terms
    for p, q, c in one_body_terms:
        if not (0 <= int(p) < num_spin_orbitals and 0 <= int(q) < num_spin_orbitals):
            raise ValueError("one_body_terms contain orbital indices outside num_spin_orbitals")
        add_term(f"+_{int(p)} -_{int(q)}", float(c))
        if int(p) != int(q):
            # add Hermitian conjugate assuming real coefficients
            add_term(f"+_{int(q)} -_{int(p)}", float(c))

    # two-body terms (optional)
    if two_body_terms is not None:
        for p, q, r, s, c in two_body_terms:
            if not (
                0 <= int(p) < num_spin_orbitals
                and 0 <= int(q) < num_spin_orbitals
                and 0 <= int(r) < num_spin_orbitals
                and 0 <= int(s) < num_spin_orbitals
            ):
                raise ValueError("two_body_terms contain orbital indices outside num_spin_orbitals")
            add_term(f"+_{int(p)} +_{int(q)} -_{int(r)} -_{int(s)}", float(c))
            # add Hermitian conjugate assuming real coefficients
            add_term(f"+_{int(s)} +_{int(r)} -_{int(q)} -_{int(p)}", float(c))

    ferm_op = FermionicOp(term_dict, num_spin_orbitals=num_spin_orbitals)

    # select mapper
    mapper_lower = mapper.strip().lower()
    if mapper_lower in {"jordan_wigner", "jordan-wigner", "jw"}:
        qmapper = JordanWignerMapper()
    elif mapper_lower in {"parity", "parity_mapper"}:
        # try to use parity with optional two-qubit reduction if supported by the installed version
        try:
            if two_qubit_reduction:
                if num_particles is None:
                    raise ValueError("num_particles must be provided when two_qubit_reduction is True for parity mapping")
                qmapper = ParityMapper(num_particles=num_particles)  # type: ignore[arg-type]
            else:
                qmapper = ParityMapper()  # type: ignore[call-arg]
        except TypeError:
            # fallback for versions where ParityMapper has no num_particles argument
            qmapper = ParityMapper()
    elif mapper_lower in {"bksf", "bravyi-kitaev-superfast", "bravyikitaevsuperfast"}:
        qmapper = BravyiKitaevSuperFastMapper()
    else:
        raise ValueError("Unsupported mapper. Use 'jordan_wigner', 'parity', or 'bksf'.")

    qubit_op = qmapper.map(ferm_op)

    # validate target qubit count
    if qubit_op.num_qubits != int(num_qubits):
        raise ValueError(
            f"Mapped operator qubit count ({qubit_op.num_qubits}) does not match num_qubits ({num_qubits})."
        )

    # IQPE expects a Hermitian Pauli sum with real coefficients; SparsePauliOp satisfies this requirement
    return qubit_op


# === observable_verifier_fermion_IQPE.py ===
# observable-----------------------------------------------------------------------------------------
def observable_verifier_fermion_IQPE(
    num_qubits: int,
    pauli_terms: Optional[List[Tuple[str, Any]]] = None,
    fermionic_op: Optional[Any] = None,
    mapper: Optional[Any] = None,
    register_length: Optional[int] = None,
    layout: Optional[Any] = None,
    simplify: bool = True,
    drop_identity: bool = False,
) -> SparsePauliOp:
    # a verifier observable for fermion_IQPE system: build SparsePauliOp measurement operators without running algorithms
    # num_qubits: total number of qubits expected for the observable labels
    # pauli_terms: list of (label, coeff) pairs to construct SparsePauliOp directly
    # fermionic_op: fermionic operator produced upstream (effector), to be mapped to qubits
    # mapper: qiskit-nature second-quantized mapper instance used to convert fermionic_op to SparsePauliOp
    # register_length: explicit qubit register length for mapper.map when mapping fermionic_op
    # layout: transpiler layout to align observable qubit indices with an execution circuit
    # simplify: whether to combine duplicate terms and prune near-zero coefficients in the observable
    # drop_identity: whether to remove identity-only terms ("I" * num_qubits) from the observable

    if num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer")

    op: Optional[SparsePauliOp] = None

    # Build from explicit Pauli term list if provided
    if pauli_terms is not None:
        if not isinstance(pauli_terms, list) or len(pauli_terms) == 0:
            raise ValueError("pauli_terms must be a non-empty list of (label, coeff) pairs")
        for label, _ in pauli_terms:
            if not isinstance(label, str):
                raise TypeError("Each Pauli term label must be a string")
            if len(label) != num_qubits:
                raise ValueError(
                    f"Pauli label length {len(label)} does not match num_qubits {num_qubits}"
                )
        op = SparsePauliOp.from_list(pauli_terms)

    # Otherwise, map a fermionic operator to a qubit observable
    elif fermionic_op is not None:
        if mapper is None:
            raise ValueError("mapper must be provided when fermionic_op is given")
        # Use mapper to obtain a SparsePauliOp. No algorithms are executed here.
        if register_length is not None:
            mapped = mapper.map(fermionic_op, register_length=register_length)
        else:
            mapped = mapper.map(fermionic_op)
        if not isinstance(mapped, SparsePauliOp):
            raise TypeError("Mapper must return a SparsePauliOp for measurement-only workflows")
        op = mapped
        if op.num_qubits != num_qubits:
            raise ValueError(
                f"Mapped operator has {op.num_qubits} qubits but num_qubits={num_qubits} was requested"
            )

    else:
        raise ValueError("Provide either pauli_terms or fermionic_op to construct the observable")

    # Align observable with the transpiled circuit layout if supplied
    if layout is not None:
        try:
            op = op.apply_layout(layout, num_qubits=num_qubits)
        except TypeError:
            op = op.apply_layout(layout)

    # Optionally simplify to combine duplicates and drop near-zero terms
    if simplify:
        op = op.simplify()

    # Optionally remove identity-only terms
    if drop_identity:
        labels = op.paulis.to_labels()
        coeffs = op.coeffs
        filtered = [
            (label, coeff)
            for label, coeff in zip(labels, coeffs)
            if label != ("I" * num_qubits)
        ]
        op = SparsePauliOp.from_list(filtered) if filtered else SparsePauliOp.from_list([])

    return op


# === main integration ===
def main():
    # Minimal parameters for quick verification
    num_spin_orbitals = 2
    num_qubits = 2

    # Simple fermionic one-body terms (Hermitian will be enforced in builder)
    one_body_terms = [
        (0, 0, 0.6),
        (1, 1, -0.4),
        (0, 1, 0.1),  # off-diagonal hopping; conjugate will be added internally
    ]
    two_body_terms = None

    # Build qubit Hamiltonian
    qubit_op = hamiltonian_verifier_fermion_IQPE(
        num_qubits=num_qubits,
        num_spin_orbitals=num_spin_orbitals,
        one_body_terms=one_body_terms,
        two_body_terms=two_body_terms,
        mapper="jordan_wigner",
    )

    # Prepare initial state |10> for 2-qubit register
    circinit = circinit_verifier_fermion_IQPE(
        num_qubits=num_qubits,
        initial_bitstring=[1, 0],
    )

    # Build controlled evolution with a single evaluation qubit
    base_time = 0.5
    eval_qubits = 1
    circevol = circevol_verifier_fermion_IQPE(
        hamiltonian=qubit_op,
        base_time=base_time,
        eval_qubits=eval_qubits,
        evolution="Trotter",
        append_barriers=True,
    )

    # Set up Sampler primitive for IQPE
    sampler = Sampler()

    # Execute effector (IQPE)
    try:
        eff_res = effector_verifier_fermion_IQPE(
            hamiltonian=qubit_op,
            circinit=circinit,
            circevol=circevol,
            time=base_time,
            num_iterations=3,
            sampler=sampler,
        )
        print("IQPE Results:")
        print({
            "phase": eff_res.get("phase"),
            "angle": eff_res.get("angle"),
            "energy": eff_res.get("energy"),
            "num_iterations": eff_res.get("num_iterations"),
        })
    except Exception as e:
        # Tolerant error handling per instructions
        print(f"IQPE execution encountered an issue: {e}")

    # Construct a simple observable from explicit Pauli terms
    pauli_terms = [("ZI", 1.0), ("IZ", 0.5)]
    try:
        obs = observable_verifier_fermion_IQPE(
            num_qubits=num_qubits,
            pauli_terms=pauli_terms,
            drop_identity=True,
        )
        print("Observable (explicit Pauli list) qubits:", obs.num_qubits)
        print("Observable terms:", list(zip(obs.paulis.to_labels(), obs.coeffs)))
    except Exception as e:
        print(f"Observable (pauli_terms) construction issue: {e}")

    # Also demonstrate observable built from a FermionicOp via a mapper
    try:
        ferm_op_demo = FermionicOp({"+_0 -_0": 1.0, "+_1 -_1": 0.5}, num_spin_orbitals=num_spin_orbitals)
        jw_mapper = JordanWignerMapper()
        obs_mapped = observable_verifier_fermion_IQPE(
            num_qubits=num_qubits,
            fermionic_op=ferm_op_demo,
            mapper=jw_mapper,
        )
        print("Observable (mapped from FermionicOp) qubits:", obs_mapped.num_qubits)
        print("Observable mapped terms:", list(zip(obs_mapped.paulis.to_labels(), obs_mapped.coeffs)))
    except Exception as e:
        print(f"Observable (mapped) construction issue: {e}")


if __name__ == "__main__":
    main()
