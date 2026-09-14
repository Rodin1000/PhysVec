import qiskit
from qiskit.quantum_info import SparsePauliOp
import qiskit_nature
import qiskit_algorithms
import qiskit_aer
from typing import List, Tuple, Optional, Any

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
