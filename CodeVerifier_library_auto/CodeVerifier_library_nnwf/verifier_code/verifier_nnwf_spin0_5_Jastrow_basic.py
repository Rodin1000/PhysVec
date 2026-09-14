import netket as nk
import flax.linen as nn
import jax.numpy as jnp
from flax import nnx

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_Jastrow(g, *, s=0.5, total_sz=None):
    # a verifier hilbert for spin0_5_Jastrow system: creates a spin-1/2 Hilbert space for Jastrow calculations
    # g: graph object representing the lattice structure (e.g., nk.graph.Square)
    # s: spin quantum number (default: 0.5)
    # total_sz: optional total spin projection constraint (default: None)
    #
    # Creates a Hilbert space for spin-1/2 particles using the computational basis
    # where configurations are represented as arrays of +1 (spin up) and -1 (spin down)
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_Jastrow(hi, *, kernel_init=None):
    # a verifier statemodel for spin0_5_Jastrow system: creates a Jastrow ansatz model for spin-1/2 systems
    # hi: Hilbert space object defining the spin-1/2 system
    # kernel_init: initializer for the Jastrow kernel parameters (default: normal distribution)
    #
    # Creates a Jastrow model with a symmetric kernel matrix of learnable complex parameters
    # that captures pairwise spin correlations beyond mean-field
    #
    if kernel_init is None:
        kernel_init = nn.initializers.normal()
    
    ma = nk.models.Jastrow(kernel_init=kernel_init)
    
    return ma

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_Jastrow(hi, ma, *, n_chains=16, n_samples=1008, learning_rate=0.01):
    # a verifier compset for spin0_5_Jastrow system: creates sampler, optimizer, and variational state for Jastrow wavefunction
    # hi: Hilbert space object for spin-1/2 system
    # ma: model/ansatz (Jastrow wavefunction)
    # n_chains: number of parallel MCMC chains (default: 16)
    # n_samples: number of samples per iteration (default: 1008)
    # learning_rate: learning rate for optimizer (default: 0.01)
    #    
    # Create MetropolisLocal sampler for spin flips
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=n_chains)
    
    # Define optimizer (SGD with specified learning rate)
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # Construct variational quantum state with sampler and model
    vs = nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples)
    
    return sa, op, vs

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_Jastrow(hi, g):
    # a verifier hamiltonian for spin0_5_Jastrow system: creates a Heisenberg Hamiltonian for spin-1/2 system with zero magnetization
    # hi: Hilbert space object for spin-1/2 degrees of freedom with zero magnetization constraint
    # g: graph object representing the lattice structure
    #
    # Creates a Heisenberg Hamiltonian using the provided Hilbert space and graph
    # The Hamiltonian captures short-range spin correlations as in the Jastrow model
    #
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
    return ha

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_Jastrow(ha, op, vs):
    # a verifier effector for spin0_5_Jastrow system: sets up and runs variational Monte Carlo calculation
    # ha: Hamiltonian operator for the system
    # op: Optimizer for the variational parameters
    # vs: Variational state containing the model and sampler
    #
    # Create VMC driver with Hamiltonian, optimizer, and variational state
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    # Run the optimization for a set number of iterations
    driv.run(n_iter=300)
    # Return the driver object containing the calculation results
    return driv

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_Jastrow(hi, *, indices=None):
    # a verifier observable for spin0_5_Jastrow system: defines observables for spin-1/2 Jastrow wave function calculations
    # hi: Hilbert space object for spin-1/2 system
    # indices: list of site indices to define local observables (default: all sites)
    #
    # Create magnetization observables as example observables
    if indices is None:
        indices = list(range(hi.size))
    
    # Example observable: total magnetization in x direction
    obs = nk.operator.spin.sigmax(hi, indices[0])
    for i in indices[1:]:
        obs += nk.operator.spin.sigmax(hi, i)
    
    return obs

def main():
    # Define minimal system parameters for quick execution
    # Use a small 4-site chain graph
    g = nk.graph.Chain(length=4)
    
    # Create Hilbert space for spin-1/2 system
    hi = hilbert_verifier_spin0_5_Jastrow(g, s=0.5, total_sz=0.0)
    
    # Create Jastrow ansatz model
    ma = statemodel_verifier_spin0_5_Jastrow(hi)
    
    # Create sampler, optimizer, and variational state
    sa, op, vs = compset_verifier_spin0_5_Jastrow(hi, ma, n_chains=4, n_samples=100, learning_rate=0.01)
    
    # Create Hamiltonian
    ha = hamiltonian_verifier_spin0_5_Jastrow(hi, g)
    
    # Run VMC calculation
    driv = effector_verifier_spin0_5_Jastrow(ha, op, vs)
    
    # Compute observables
    obs = observable_verifier_spin0_5_Jastrow(hi)
    
    # Compute expectation value of observable
    obs_result = vs.expect(obs)
    
    print(f"Energy: {vs.expect(ha)}")
    print(f"Observable (sigmax): {obs_result}")
    
    return {
        'hilbert': hi,
        'model': ma,
        'sampler': sa,
        'optimizer': op,
        'variational_state': vs,
        'hamiltonian': ha,
        'driver': driv,
        'observable': obs,
        'observable_result': obs_result
    }

if __name__ == "__main__":
    results = main()
