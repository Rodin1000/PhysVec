import qiskit
from typing import Any, Dict, List, Optional, Tuple, Union
from qiskit.quantum_info import SparsePauliOp, PauliList
from qiskit.opflow import PauliSumOp
import qiskit_aer
from qiskit_algorithms import VQE  # imported as required, not used here
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import QubitMapper


# observable-----------------------------------------------------------------------------------------
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
                    prepared[name] = [PauliSumOp(g) for g in groups]
                else:
                    prepared[name] = groups
            else:
                prepared[name] = PauliSumOp(op) if output_type == "PauliSumOp" else op

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
                    prepared[name] = [PauliSumOp(g) for g in groups]
                else:
                    prepared[name] = groups
            else:
                prepared[name] = PauliSumOp(qubit_op) if output_type == "PauliSumOp" else qubit_op

    return prepared
