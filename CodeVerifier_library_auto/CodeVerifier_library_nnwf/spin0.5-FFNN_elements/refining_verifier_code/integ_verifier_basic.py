import netket as nk
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_FFNN(g, *, s: float = 0.5):
    # a verifier hilbert for spin0_5_FFNN system: creates a Hilbert space for spin-1/2 system with given graph
    # g: graph object representing the lattice structure, must have n_nodes attribute
    # s: spin value, default is 0.5 for spin-1/2 systems
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes)
    #
    return hi

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_FFNN(*, features_factor: int = 2, use_bias: bool = True, param_dtype: type = np.complex128, kernel_stddev: float = 0.01, bias_stddev: float = 0.01):
    # a verifier statemodel for spin0_5_FFNN system: defines a feed-forward neural network model for spin 1/2 systems using flax.linen Module
    # features_factor: multiplier for the number of features in the Dense layer, calculated as features_factor * x.shape[-1] (default: 2)
    # use_bias: whether to use bias in the Dense layer (default: True)
    # param_dtype: data type for parameters (default: np.complex128)
    # kernel_stddev: standard deviation for kernel initialization (default: 0.01)
    # bias_stddev: standard deviation for bias initialization (default: 0.01)
    
    class FFNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Apply Dense layer with features = features_factor * x.shape[-1]
            x = nn.Dense(
                features=features_factor * x.shape[-1],
                use_bias=use_bias,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(kernel_stddev),
                bias_init=nn.initializers.normal(bias_stddev)
            )(x)
            # Apply log_cosh activation
            x = jnp.log(jnp.cosh(x))
            # Sum over the last axis
            return jnp.sum(x, axis=-1)
    
    # Instantiate and return the model object
    ma = FFNN()
    return ma

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_FFNN(hi, g, model, *, n_samples: int = 1008, learning_rate: float = 0.01):
    # a verifier compset for spin0.5_FFNN system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for spin-1/2 system
    # g: graph/lattice object defining the system geometry
    # model: variational model (ansatz) such as RBM or FFNN
    # n_samples: number of Monte Carlo samples for gradient estimation (default: 1008)
    # learning_rate: learning rate for the optimizer (default: 0.01)
    
    # Create sampler for spin-1/2 system
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)
    
    # Create optimizer for variational parameters
    op = nk.optimizer.Adam(learning_rate=learning_rate)
    
    # Create variational quantum state
    vs = nk.vqs.MCState(sa, model, n_samples=n_samples)
    
    return sa, op, vs

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_FFNN(hi, g):
    # a verifier hamiltonian for spin0_5_FFNN system: creates a Heisenberg Hamiltonian for spin-1/2 system
    # hi: Hilbert space object for spin-1/2 degrees of freedom
    # g: graph object defining lattice structure and connectivity
    #
    # Creates a Heisenberg Hamiltonian using NetKet's built-in operator
    #
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
    return ha

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_FFNN(vs, ha, optimizer):
    # a verifier effector for spin0_5_FFNN system: creates and returns a VMC driver object for spin-1/2 system
    # vs: variational state object (e.g., MCState) containing the neural network model and Hilbert space
    # ha: Hamiltonian operator (e.g., LocalOperator) defining the system's energy
    # op: optimizer (e.g., optax optimizer) for variational parameter updates
    #
    # Create VMC driver with variational state, Hamiltonian, and optimizer
    driv = nk.driver.VMC(hamiltonian=ha, variational_state=vs, optimizer=optimizer)
    driv.diag_shift = 0.01
    #
    return driv

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_FFNN(hi, *, dtype=complex):
    # a verifier observable for spin0_5_FFNN system: defines an observable operator for spin0.5-FFNN system
    # hi: Hilbert space object defining the quantum state space for the spin0.5 system
    # dtype: data type of the observable's output (default: complex)
    
    # Create observable as a dictionary containing common observables
    obs = {
        "energy": nk.operator.spin.sigmax(hi, 0),  # Example observable: sigma_x on first site
        "magnetization": nk.operator.spin.sigmaz(hi, 0)  # Example observable: sigma_z on first site
    }
    
    return obs

def main():
    # Create a simple 1D chain graph with 4 sites (small system size for quick execution)
    g = nk.graph.Hypercube(length=4, n_dim=1)
    
    # Create Hilbert space for spin-1/2 system
    hi = hilbert_verifier_spin0_5_FFNN(g)
    
    # Create the FFNN model
    model = statemodel_verifier_spin0_5_FFNN()
    
    # Create sampler, optimizer, and variational state
    sa, op, vs = compset_verifier_spin0_5_FFNN(hi, g, model, n_samples=100, learning_rate=0.01)
    
    # Create Hamiltonian
    ha = hamiltonian_verifier_spin0_5_FFNN(hi, g)
    
    # Create VMC driver
    driv = effector_verifier_spin0_5_FFNN(vs, ha, op)
    
    # Create observables
    obs = observable_verifier_spin0_5_FFNN(hi)
    
    # Return all objects for potential further use
    return {
        'graph': g,
        'hilbert': hi,
        'model': model,
        'sampler': sa,
        'optimizer': op,
        'variational_state': vs,
        'hamiltonian': ha,
        'driver': driv,
        'observables': obs
    }

# Call main function if script is run directly
if __name__ == "__main__":
    results = main()
    print("Integrated verifier program executed successfully.")
    print(f"Number of parameters in model: {results['variational_state'].n_parameters}")
    print(f"Hilbert space size: {results['hilbert'].n_states}")