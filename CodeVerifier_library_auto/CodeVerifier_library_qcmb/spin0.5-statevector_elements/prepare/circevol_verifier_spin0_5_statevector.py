from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import PauliSumOp
from qiskit.synthesis import LieTrotter

# circevol-----------------------------------------------------------------------------------------
def circevol_verifier_spin0_5_statevector(hamiltonian: PauliSumOp, time: float = 1.0, num_timesteps: int = 1, synthesis: str = "lie_trotter"):
    # a verifier circevol for spin0_5_statevector system: creates a quantum circuit for time evolution using Trotterization
    # hamiltonian: PauliSumOp representing the system Hamiltonian (required)
    # time: evolution time parameter (default: 1.0)
    # num_timesteps: number of Trotter steps (default: 1)
    # synthesis: type of Trotter synthesis method ("lie_trotter" or "suzuki") (default: "lie_trotter")
    
    # Create evolution gate with specified synthesis method
    if synthesis == "lie_trotter":
        synthesis_method = LieTrotter(reps=num_timesteps)
    else:
        # For simplicity, default to LieTrotter if other methods not specified
        synthesis_method = LieTrotter(reps=num_timesteps)
    
    # Create the evolution gate
    evolution_gate = PauliEvolutionGate(hamiltonian, time, synthesis=synthesis_method)
    
    # Create quantum circuit with same number of qubits as hamiltonian
    num_qubits = hamiltonian.num_qubits
    qc = QuantumCircuit(num_qubits)
    
    # Append the evolution gate to the circuit
    qc.append(evolution_gate, range(num_qubits))
    
    return qc