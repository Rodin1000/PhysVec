import math
import qiskit
import qiskit_nature
import qiskit_aer
from typing import Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_algorithms.phase_estimators import IterativePhaseEstimation
from qiskit_aer.primitives import Sampler

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
