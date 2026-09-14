from typing import Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit_aer import Aer  # required explicit import per format
from qiskit_algorithms import VQE  # required explicit import per format
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import QubitMapper


# circinit-----------------------------------------------------------------------------------------
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
