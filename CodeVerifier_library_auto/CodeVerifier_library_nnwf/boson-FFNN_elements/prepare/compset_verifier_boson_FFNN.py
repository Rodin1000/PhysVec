import netket as nk

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
