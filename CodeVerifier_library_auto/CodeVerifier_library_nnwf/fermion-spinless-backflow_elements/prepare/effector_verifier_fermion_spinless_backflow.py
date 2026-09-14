import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_fermion_spinless_backflow(ha, op, vs):
    # a verifier effector for fermion_spinless_backflow system: creates a VMC driver for optimizing a neural network wave function with backflow transformations
    # ha: Hamiltonian operator for the system
    # op: Optimizer for parameter updates (e.g., nk.optimizer.Adam)
    # vs: Variational state containing the neural network model with backflow transformations
    #
    # Creates a VMC driver object that orchestrates the variational Monte Carlo optimization
    # for a fermionic system with spinless backflow wave functions.
    #
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    return driv