import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_boson_FFNN(ha, op, vs, *, diag_shift: float = 0.01):
    # a verifier effector for boson_FFNN system: creates VMC driver for ground state optimization
    # ha: Hamiltonian operator for the boson system
    # op: optimizer object (e.g., nk.optimizer.Sgd)
    # vs: variational state (MCState) containing sampler and neural network model
    # diag_shift: regularization parameter for stochastic reconfiguration (default 0.01)
    
    driv = nk.driver.VMC_SR(hamiltonian=ha, optimizer=op, variational_state=vs, diag_shift=diag_shift)
    
    return driv
