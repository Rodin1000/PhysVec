import netket as nk

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_tVMC(hi, ma, *, optimizer_name="Adam", n_samples=1000):
    # a verifier compset for spin0_5_tVMC system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for spin-1/2 system
    # ma: variational model/ansatz (e.g., neural network model like RBM)
    # optimizer_name: name of optimizer to use (default "Adam")
    # n_samples: number of Monte Carlo samples to use (default 1000)
    
    # Create sampler using MetropolisLocal which flips spins one by one
    sa = nk.sampler.MetropolisLocal(hi)
    
    # Create variational quantum state using the sampler and model
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    # Create optimizer based on specified name
    if optimizer_name == "Adam":
        op = nk.optimizer.Adam()
    elif optimizer_name == "Sgd":
        op = nk.optimizer.Sgd()
    elif optimizer_name == "Momentum":
        op = nk.optimizer.Momentum()
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    return sa, op, vs