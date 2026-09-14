import qiskit
import qiskit_nature
import qiskit_algorithms
import qiskit_aer
from typing import Dict, Tuple, Optional
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, BravyiKitaevMapper

# hamiltonian-----------------------------------------------------------------------------------------
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
