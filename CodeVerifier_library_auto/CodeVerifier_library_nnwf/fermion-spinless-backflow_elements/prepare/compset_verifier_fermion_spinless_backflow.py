import netket as nk

# compset-----------------------------------------------------------------------------------------
def compset_verifier_fermion_spinless_backflow(hi, graph, model, learning_rate=0.05, n_samples=4096, n_discard_per_chain=16):
    # a verifier compset for fermion_spinless_backflow system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for fermionic system (SpinOrbitalFermions)
    # graph: lattice/graph object defining the system geometry
    # model: neural network model defining the wave function (e.g., LogNeuralBackflow)
    # learning_rate: learning rate for the optimizer (default: 0.05)
    # n_samples: number of Monte Carlo samples per iteration (default: 4096)
    # n_discard_per_chain: number of samples to discard for thermalization (default: 16)
    
    # Create MetropolisFermionHop sampler to conserve particle number in fermionic system
    sa = nk.sampler.MetropolisFermionHop(hi, graph=graph)
    
    # Create optimizer for parameter updates
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # Create variational quantum state combining sampler and model
    vs = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain)
    
    return sa, op, vs