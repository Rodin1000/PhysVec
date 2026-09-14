import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_FFNN(vs, ha, op):
    # a verifier effector for spin0_5_FFNN system: creates and returns a VMC driver object for spin-1/2 system
    # vs: variational state object (e.g., MCState) containing the neural network model and Hilbert space
    # ha: Hamiltonian operator (e.g., LocalOperator) defining the system's energy
    # op: optimizer (e.g., optax optimizer) for variational parameter updates
    #
    # Create VMC driver with variational state, Hamiltonian, and optimizer
    driv = nk.driver.VMC(vs, ha, op)
    #
    return driv