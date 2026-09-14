import netket as nk

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin, ma, *, n_samples: int = 1024, learning_rate: float = 0.01):
    # a verifier compset for spin0_5_tVMC system: defines the sampler, optimizer, and variational state
    # hi: the Hilbert space object for the spin 0.5 system
    # ma: the variational model or ansatz (e.g., RBM)
    # n_samples: number of Monte Carlo samples to use in the variational state (default: 1024)
    # learning_rate: the learning rate for the optimizer (default: 0.01)

    # Define the sampler
    sa = nk.sampler.MetropolisLocal(hi)

    # Define the optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)

    # Define the variational quantum state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)

    return sa, op, vs
