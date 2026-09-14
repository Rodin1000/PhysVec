import netket as nk

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