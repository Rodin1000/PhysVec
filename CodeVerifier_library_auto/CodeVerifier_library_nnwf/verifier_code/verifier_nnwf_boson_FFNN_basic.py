# === Integrated Verifier for boson_FFNN ===
# Consolidated imports
import netket as nk
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import numpy as np


# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_boson_FFNN(N: int, n_max: int, *, n_particles: int = None):
    # a verifier hilbert for boson_FFNN system: defines a Fock (boson) Hilbert space with specified sites and max occupation
    # N: number of bosonic modes/sites in the Hilbert space
    # n_max: maximum occupation number per site (inclusive), determines local dimension as (n_max + 1)
    # n_particles: optional constraint for total number of particles (default None means no constraint)
    
    hi = nk.hilbert.Fock(n_max=n_max, N=N, n_particles=n_particles)
    
    return hi


# statemodel-----------------------------------------------------------------------------------------
class FFNN(nn.Module):
    # feed-forward neural network model for boson systems
    # alpha: feature density (hidden layer size = alpha * input size)
    alpha: int = 1
    
    @nn.compact
    def __call__(self, x):
        # x: input configuration with shape (n_samples, N)
        # apply dense layer with complex parameters
        x = nn.Dense(features=self.alpha * x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128,
                     kernel_init=nn.initializers.normal(stddev=0.01))(x)
        # apply activation function
        x = nknn.log_cosh(x)
        # sum over features to get scalar log-amplitude
        return jnp.sum(x, axis=-1)

def statemodel_verifier_boson_FFNN(alpha: int = 1):
    # a verifier statemodel for boson_FFNN system: creates a feed-forward neural network model
    # alpha: feature density determining hidden layer size as alpha * input_size (default 1)
    #
    # create FFNN model instance with specified alpha
    ma = FFNN(alpha=alpha)
    #
    return ma


# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_boson_FFNN(hi, g, *, U: float = 1.0, V: float = 0.0, J: float = 1.0, mu: float = 0.0):
    # a verifier hamiltonian for boson_FFNN system: Bose-Hubbard Hamiltonian with on-site and density-density interactions
    # hi: Fock Hilbert space object for bosons
    # g: graph/lattice object defining the connectivity
    # U: on-site interaction strength (default 1.0)
    # V: density-density interaction strength between neighboring sites (default 0.0)
    # J: hopping amplitude between neighboring sites (default 1.0)
    # mu: chemical potential (default 0.0)
    
    ha = nk.operator.BoseHubbard(hilbert=hi, U=U, V=V, J=J, mu=mu, graph=g)
    
    return ha


# compset-----------------------------------------------------------------------------------------
def compset_verifier_boson_FFNN(hi, ma, *, n_samples: int = 1008, n_chains: int = 16, learning_rate: float = 0.01):
    # a verifier compset for boson_FFNN system: creates sampler, optimizer, and variational state for VMC
    # hi: Hilbert space object (e.g., nk.hilbert.Fock)
    # ma: variational model / neural network ansatz (e.g., FFNN defined with Flax)
    # n_samples: total number of samples across all chains (default 1008)
    # n_chains: number of independent Markov chains for sampling (default 16)
    # learning_rate: learning rate for the optimizer (default 0.01)
    
    # create sampler for bosonic Hilbert space
    sa = nk.sampler.MetropolisLocal(hi, n_chains=n_chains)
    
    # create optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # create variational quantum state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    return sa, op, vs


# effector-----------------------------------------------------------------------------------------
def effector_verifier_boson_FFNN(ha, op, vs, *, diag_shift: float = 0.01):
    # a verifier effector for boson_FFNN system: creates VMC driver for ground state optimization
    # ha: Hamiltonian operator for the boson system
    # op: optimizer object (e.g., nk.optimizer.Sgd)
    # vs: variational state (MCState) containing sampler and neural network model
    # diag_shift: regularization parameter for stochastic reconfiguration (default 0.01)
    
    driv = nk.driver.VMC_SR(hamiltonian=ha, optimizer=op, variational_state=vs, diag_shift=diag_shift)
    
    return driv


# observable-----------------------------------------------------------------------------------------
def observable_verifier_boson_FFNN(hi, sites: list = None, *, return_dict: bool = False):
    # a verifier observable for boson_FFNN system: defines boson number observables for VMC calculation
    # hi: Hilbert space object (nk.hilbert.Fock) for the boson system
    # sites: list of site indices to measure number operator (default: all sites)
    # return_dict: if True, return dictionary of observables; if False, return total number operator (default: False)
    
    # Get number of sites from Hilbert space
    N = hi.size
    
    # Default to all sites if not specified
    if sites is None:
        sites = list(range(N))
    
    if return_dict:
        # Create dictionary of observables for individual site occupations
        obs = {}
        for i in sites:
            obs[f'n_{i}'] = nk.operator.boson.number(hi, i)
        # Add total particle number
        obs['n_total'] = sum([nk.operator.boson.number(hi, i) for i in sites])
    else:
        # Return total number operator as single observable
        obs = sum([nk.operator.boson.number(hi, i) for i in sites])
    
    return obs


# main-----------------------------------------------------------------------------------------
def main():
    """Main function that chains all element functions for boson_FFNN verification."""
    
    # Define minimal parameters for quick verification
    N = 4              # number of sites (small for quick execution)
    n_max = 2          # maximum occupation per site
    n_particles = 4    # total number of particles (optional constraint)
    alpha = 1          # feature density for FFNN
    
    # Bose-Hubbard parameters
    U = 1.0            # on-site interaction
    V = 0.0            # density-density interaction
    J = 1.0            # hopping amplitude
    mu = 0.0           # chemical potential
    
    # VMC parameters
    n_samples = 512    # number of samples (small for quick execution)
    n_chains = 8       # number of Markov chains
    learning_rate = 0.01
    diag_shift = 0.01
    
    # Step 1: Create Hilbert space
    print("Creating Hilbert space...")
    hi = hilbert_verifier_boson_FFNN(N=N, n_max=n_max, n_particles=n_particles)
    print(f"  Hilbert space: {hi}")
    
    # Step 2: Create neural network model (FFNN)
    print("Creating FFNN model...")
    ma = statemodel_verifier_boson_FFNN(alpha=alpha)
    print(f"  Model: {ma}")
    
    # Step 3: Create graph/lattice for Hamiltonian
    print("Creating lattice graph...")
    g = nk.graph.Chain(length=N, pbc=True)
    print(f"  Graph: {g}")
    
    # Step 4: Create Hamiltonian (Bose-Hubbard)
    print("Creating Hamiltonian...")
    ha = hamiltonian_verifier_boson_FFNN(hi, g, U=U, V=V, J=J, mu=mu)
    print(f"  Hamiltonian: {ha}")
    
    # Step 5: Create computational setup (sampler, optimizer, variational state)
    print("Creating computational setup...")
    sa, op, vs = compset_verifier_boson_FFNN(hi, ma, n_samples=n_samples, n_chains=n_chains, learning_rate=learning_rate)
    print(f"  Sampler: {sa}")
    print(f"  Optimizer: {op}")
    print(f"  Variational state: {vs}")
    
    # Step 6: Create VMC driver (effector)
    print("Creating VMC driver...")
    driv = effector_verifier_boson_FFNN(ha, op, vs, diag_shift=diag_shift)
    print(f"  Driver: {driv}")
    
    # Step 7: Create observables
    print("Creating observables...")
    obs = observable_verifier_boson_FFNN(hi, return_dict=True)
    print(f"  Observables: {list(obs.keys())}")
    
    # Run a few VMC iterations to verify everything works
    print("\nRunning VMC optimization (2 iterations)...")
    driv.run(n_iter=2, obs=obs)
    
    print("\n=== Verification Complete ===")
    print(f"Final energy: {vs.expect(ha)}")
    
    return hi, ma, g, ha, sa, op, vs, driv, obs


if __name__ == "__main__":
    main()
