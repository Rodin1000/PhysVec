import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_boson_RBM(ha, op, vs):
    # a verifier effector for boson_RBM system: creates VMC driver for variational Monte Carlo optimization
    # ha: Hamiltonian operator defining the problem (e.g., nk.operator.LocalOperator or nk.operator.Ising)
    # op: optimizer for energy minimization (e.g., nk.optimizer.SGD or nk.optimizer.Sgd)
    # vs: variational state object (e.g., nk.vqs.MCState with RBM model)
    
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    
    return driv
