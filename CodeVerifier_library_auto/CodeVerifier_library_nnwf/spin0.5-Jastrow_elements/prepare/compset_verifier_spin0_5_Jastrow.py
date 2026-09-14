import netket as nk
import flax.linen as nn
import jax.numpy as jnp
from flax import nnx

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_Jastrow(hi, ma, *, n_chains=16, n_samples=1008, learning_rate=0.01):
    # a verifier compset for spin0_5_Jastrow system: creates sampler, optimizer, and variational state for Jastrow wavefunction
    # hi: Hilbert space object for spin-1/2 system
    # ma: model/ansatz (Jastrow wavefunction)
    # n_chains: number of parallel MCMC chains (default: 16)
    # n_samples: number of samples per iteration (default: 1008)
    # learning_rate: learning rate for optimizer (default: 0.01)
    
    # Create MetropolisLocal sampler for spin flips
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=n_chains)
    
    # Define optimizer (SGD with specified learning rate)
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # Construct variational quantum state with sampler and model
    vs = nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples)
    
    return sa, op, vs