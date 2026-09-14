import netket as nk

# compset-----------------------------------------------------------------------------------------
def compset_verifier_boson_RBM(hi, ma, *, n_samples: int = 1008, n_chains: int = 16, sweep_size: int = 100, learning_rate: float = 0.01):
    # a verifier compset for boson_RBM system: creates sampler, optimizer, and variational state for VMC
    # hi: Hilbert space object (e.g., nk.hilbert.Fock)
    # ma: variational model / ansatz (e.g., nk.models.RBM)
    # n_samples: total number of samples for MCState (default 1008)
    # n_chains: number of Markov chains for sampler (default 16)
    # sweep_size: number of steps per sweep in sampler (default 100)
    # learning_rate: learning rate for SGD optimizer (default 0.01)
    
    # create sampler with MetropolisLocal for local updates
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=n_chains, sweep_size=sweep_size)
    
    # create optimizer with SGD
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # create variational state MCState with sampler and model
    vs = nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples)
    
    return sa, op, vs
