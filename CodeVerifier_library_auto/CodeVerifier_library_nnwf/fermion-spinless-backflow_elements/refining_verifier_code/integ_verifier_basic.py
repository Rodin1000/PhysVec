import netket as nk
from netket.operator.fermion import destroy as c, create as cdag, number as nc
import jax.numpy as jnp
import flax.linen as nn

# === compset_verifier_fermion_spinless_backflow.py ===

# compset-----------------------------------------------------------------------------------------
def compset_verifier_fermion_spinless_backflow(hi, graph, model, learning_rate=0.05, n_samples=4096, n_discard_per_chain=16):
    # a verifier compset for fermion_spinless_backflow system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for fermionic system (SpinOrbitalFermions)
    # graph: lattice/graph object defining the system geometry
    # model: neural network model defining the wave function (e.g., LogNeuralBackflow)
    # learning_rate: learning rate for the optimizer (default: 0.05)
    # n_samples: number of Monte Carlo samples per iteration (default: 4096)
    # n_discard_per_chain: number of samples to discard for thermalization (default: 16)
    
    # Create MetropolisFermionHop sampler to conserve particle number in fermionic system
    sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
    
    # Create optimizer for parameter updates
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # Create variational quantum state combining sampler and model
    vs = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    
    return sa, op, vs

# === effector_verifier_fermion_spinless_backflow.py ===

# effector-----------------------------------------------------------------------------------------
def effector_verifier_fermion_spinless_backflow(ha, op, vs):
    # a verifier effector for fermion_spinless_backflow system: creates a VMC driver for optimizing a neural network wave function with backflow transformations
    # ha: Hamiltonian operator for the system
    # op: Optimizer for parameter updates (e.g., nk.optimizer.Adam)
    # vs: Variational state containing the neural network model with backflow transformations
    #
    # Creates a VMC driver object that orchestrates the variational Monte Carlo optimization
    # for a fermionic system with spinless backflow wave functions.
    #
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    return driv

# === hamiltonian_verifier_fermion_spinless_backflow.py ===

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_fermion_spinless_backflow(hi, graph, t, V):
    # a verifier hamiltonian for fermion_spinless_backflow system: constructs tight-binding hopping and density-density interaction terms
    # hi: Hilbert space object for the fermion-spinless-backflow system
    # graph: lattice graph object defining the system geometry
    # t: hopping parameter for tight-binding term
    # V: interaction parameter for density-density term
    #
    # Construct Hamiltonian using fermionic operators
    ha = 0.0
    for i, j in graph.edges():
        # Add hopping term
        ha -= t * (cdag(hi, i) @ c(hi, j) + cdag(hi, j) @ c(hi, i))
        # Add density-density interaction term
        ha += V * nc(hi, i) @ nc(hi, j)
    #
    return ha

# === hilbert_verifier_fermion_spinless_backflow.py ===

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_fermion_spinless_backflow(N: int, N_f: int):
    # a verifier hilbert for fermion_spinless_backflow system: creates a Hilbert space for spinless fermions with fixed particle number
    # N: number of sites/orbitals in the lattice
    # N_f: number of fermions in the system (particle number)
    #
    # Creates a Hilbert space for spinless fermions using SpinOrbitalFermions with s=None
    # and fixed particle number conservation via n_fermions parameter
    #
    hi = nk.hilbert.SpinOrbitalFermions(N, s=None, n_fermions=N_f)
    return hi

# === observable_verifier_fermion_spinless_backflow.py ===

# observable-----------------------------------------------------------------------------------------
def observable_verifier_fermion_spinless_backflow(hi, *, indices=None, observable_type="density"):
    # a verifier observable for fermion_spinless_backflow system: defines fermionic observables for spinless fermion systems
    # hi: Hilbert space object for the fermionic system
    # indices: list of site indices where the observable is defined (default: all sites)
    # observable_type: type of observable to create ("density", "correlation", or "custom") (default: "density")
    
    # Default to all sites if indices not specified
    if indices is None:
        indices = list(range(hi.size))
    
    # Create observable based on type
    if observable_type == "density":
        # Density observable: n_i = c^\dagger_i c_i
        obs = nk.operator.fermion.number(hi, site_index=indices[0]) if len(indices) == 1 else \
              sum(nk.operator.fermion.number(hi, site_index=i) for i in indices) / len(indices)
    
    elif observable_type == "correlation":
        # Correlation observable: <c^\dagger_i c_j> for i != j
        if len(indices) != 2:
            raise ValueError("Correlation observable requires exactly two site indices")
        i, j = indices
        obs = nk.operator.fermion.destroy(hi, site_index=i) @ nk.operator.fermion.create(hi, site_index=j)
    
    elif observable_type == "custom":
        # Custom observable can be extended as needed
        # Initialize as zero operator
        obs = nk.operator.fermion.zero(hi)
    
    else:
        raise ValueError(f"Unknown observable type: {observable_type}")
    
    return obs

# === statemodel_verifier_fermion_spinless_backflow.py ===

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_fermion_spinless_backflow(hilbert, N, Nf, hidden_units=16, *, param_dtype=complex, seed=1234):
    # a verifier statemodel for fermion_spinless_backflow system: defines a neural backflow wave function ansatz
    # hilbert: Hilbert space object for the fermionic system
    # N: number of sites in the lattice
    # Nf: number of fermions (number of occupied orbitals)
    # hidden_units: number of neurons in the hidden layer of the backflow network (default: 16)
    # param_dtype: data type for model parameters (default: complex)
    # seed: random seed for parameter initialization (default: 1234)
    
    class LogNeuralBackflow(nn.Module):
        N: int
        Nf: int
        hidden_units: int
        param_dtype: type = complex
        
        @nn.compact
        def __call__(self, n):
            # n: occupation numbers (batched), shape (..., N)
            
            # Initialize the base orbital matrix M
            M = self.param('M',
                          nn.initializers.normal(stddev=0.1),
                          (self.N, self.Nf),
                          self.param_dtype)
            
            # Define the backflow neural network
            F = nn.Sequential([
                nn.Dense(self.hidden_units, dtype=self.param_dtype),
                nn.tanh,
                nn.Dense(self.N * self.Nf, dtype=self.param_dtype),
                lambda x: x.reshape(x.shape[:-1] + (self.N, self.Nf))
            ])(n)
            
            # Add backflow correction to the orbital matrix
            orbitals = M + F  # shape: (..., N, Nf)
            
            # Find occupied sites
            occupied = jnp.nonzero(n, size=self.Nf)[0]  # indices of shape (Nf,)
            
            # Expand dimensions of occupied indices to match orbitals
            # occupied: (Nf,) -> (1, Nf, 1) to broadcast with orbitals (..., N, Nf)
            occupied_indices = occupied[None, :, None]  # Add batch and feature dimensions
            
            # Expand orbitals to ensure compatible dimensions
            # For single configuration, add batch dimension if not present
            if orbitals.ndim == 2:
                orbitals_expanded = orbitals[None, ...]  # Add batch dimension
            else:
                orbitals_expanded = orbitals
            
            # Use jnp.take_along_axis with properly shaped indices
            # Take along axis -2 (spatial axis) to select occupied orbitals
            # This creates shape (..., Nf, Nf) directly
            occupied_orbitals = jnp.take_along_axis(orbitals_expanded, occupied_indices, axis=-2)
            
            # Compute log determinant
            # occupied_orbitals has shape (..., Nf, Nf) after indexing
            # No need to squeeze as the shape is already correct
            log_psi = jnp.linalg.slogdet(occupied_orbitals)[1]
            
            return log_psi  # Return log determinant directly
    
    # Create model instance
    model = LogNeuralBackflow(N=N, Nf=Nf, hidden_units=hidden_units, param_dtype=param_dtype)
    
    # Create random number generators
    rngs = nk.jax.PRNGSeq(seed)
    
    # Initialize the model parameters
    variables = model.init(next(rngs), jnp.zeros((N,)))
    
    return model


def main():
    # Minimal parameters for quick verification
    N = 4  # Number of sites (small system size)
    N_f = 2  # Number of fermions
    t = 1.0  # Hopping parameter
    V = 0.5  # Interaction parameter
    learning_rate = 0.05
    n_samples = 1024  # Reduced sample size for faster execution
    n_discard_per_chain = 8  # Reduced discard samples
    hidden_units = 8  # Reduced hidden units for faster computation
    
    # Create graph (1D chain with periodic boundary conditions)
    graph = nk.graph.Chain(length=N, pbc=True)
    
    # Step 1: Create Hilbert space
    hi = hilbert_verifier_fermion_spinless_backflow(N, N_f)
    
    # Step 2: Create Hamiltonian
    ha = hamiltonian_verifier_fermion_spinless_backflow(hi, graph, t, V)
    
    # Step 3: Create model (statemodel)
    model = statemodel_verifier_fermion_spinless_backflow(hi, N, N_f, hidden_units=hidden_units)
    
    # Step 4: Create compset (sampler, optimizer, variational state)
    sa, op, vs = compset_verifier_fermion_spinless_backflow(hi, graph, model, 
                                                           learning_rate=learning_rate, 
                                                           n_samples=n_samples, 
                                                           n_discard_per_chain=n_discard_per_chain)
    
    # Step 5: Create effector (VMC driver)
    driv = effector_verifier_fermion_spinless_backflow(ha, op, vs)
    
    # Step 6: Create observable (density at site 0)
    obs = observable_verifier_fermion_spinless_backflow(hi, indices=[0], observable_type="density")
    
    # Print confirmation of successful integration
    print("All element functions successfully integrated and chained.")
    print(f"System: {N} sites, {N_f} fermions")
    print(f"Hamiltonian terms: hopping={t}, interaction={V}")
    print(f"Samples: {n_samples}, hidden units: {hidden_units}")
    
    return driv, obs  # Return driver and observable for potential further use

if __name__ == "__main__":
    main()