import netket as nk

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_RBM(hi, ma, *, n_samples: int = 1000, learning_rate: float = 0.01):
    # a verifier compset for spin0_5_RBM system: creates sampler, optimizer, and variational quantum state for spin0.5-RBM
    # hi: Hilbert space object for spin-1/2 system
    # ma: model/ansatz object (e.g., RBM)
    # n_samples: number of Monte Carlo samples for variational state (default: 1000)
    # learning_rate: learning rate for optimizer (default: 0.01)
    
    # Create sampler for spin0.5-RBM system
    sa = nk.sampler.MetropolisLocal(hi)
    
    # Construct variational quantum state with sampler and model
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    # Define optimizer for RBM wave function training
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    return sa, op, vs