from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import Optimizer

# Explicit package imports as required
import qiskit_aer  # noqa: F401
import qiskit_nature  # noqa: F401


# effector-----------------------------------------------------------------------------------------
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
        statevector = Statevector.from_instruction(optimal_circuit)

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
