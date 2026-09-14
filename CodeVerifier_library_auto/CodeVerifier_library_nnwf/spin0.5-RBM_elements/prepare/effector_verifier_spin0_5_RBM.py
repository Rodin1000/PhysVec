import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_RBM(ha, op, vs, *, n_samples=1000, n_discard=None):
    # a verifier effector for spin0_5_RBM system: creates a VMC_SR driver for spin0.5-RBM calculations
    # ha: Hamiltonian operator (e.g., nk.operator.LocalOperator)
    # op: Optimizer (e.g., nk.optimizer.Adam)
    # vs: Variational state (e.g., nk.vqs.MCState)
    # n_samples: Number of samples for Monte Carlo estimation (default: 1000)
    # n_discard: Number of samples to discard for equilibration (default: n_samples/10)
    
    # Set default value for n_discard if not provided
    if n_discard is None:
        n_discard = n_samples // 10
    
    # Update the number of samples in the variational state
    vs.n_samples = n_samples
    
    # Create VMC driver with stochastic reconfiguration
    driv = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=op,
        variational_state=vs
    )
    
    return driv