import qiskit
import qiskit_nature
import qiskit_algorithms
import qiskit_aer
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevSuperFastMapper,
)

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
