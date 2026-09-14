import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_Jastrow(ha, op, vs):
    # a verifier effector for spin0_5_Jastrow system: sets up and runs variational Monte Carlo calculation
    # ha: Hamiltonian operator for the system
    # op: Optimizer for the variational parameters
    # vs: Variational state containing the model and sampler
    #
    # Create VMC driver with Hamiltonian, optimizer, and variational state
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    # Run the optimization for a set number of iterations
    driv.run(n_iter=300)
    # Return the driver object containing the calculation results
    return driv