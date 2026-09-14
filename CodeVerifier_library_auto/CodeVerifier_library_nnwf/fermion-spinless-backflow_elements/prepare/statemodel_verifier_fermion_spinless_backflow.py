import netket as nk
import jax.numpy as jnp
import flax.linen as nn

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
            occupied = jnp.nonzero(n, size=self.Nf, keepdims=True)  # indices of occupied sites
            
            # Extract orbitals for occupied sites
            # Use advanced indexing to get the occupied orbitals
            occupied_orbitals = jnp.take_along_axis(orbitals, occupied[..., None], axis=-2)
            
            # Compute log determinant
            log_psi = jnp.linalg.slogdet(occupied_orbitals)[1]
            
            return log_psi.squeeze(-1)  # Remove the size-1 dimension from keepdims
    
    # Create model instance
    model = LogNeuralBackflow(N=N, Nf=Nf, hidden_units=hidden_units, param_dtype=param_dtype)
    
    # Create random number generators
    rngs = nk.jax.PRNGSeq(seed)
    
    # Initialize the model parameters
    variables = model.init(next(rngs), jnp.zeros((N,)))
    
    return model
