import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_tVMC(*, kernel_init=None, seed=42):
    # a verifier statemodel for spin0_5_tVMC system: returns a Jastrow ansatz model for spin-1/2 systems
    # kernel_init: initializer for the Jastrow kernel matrix (default: normal distribution)
    # seed: random seed for parameter initialization (default: 42)
    #
    # Implements a short-range Jastrow ansatz with symmetric complex kernel matrix
    # as the variational model for tVMC calculations on spin-1/2 systems
    #
    if kernel_init is None:
        kernel_init = nk.nn.initializers.normal()
    
    ma = nk.models.Jastrow(kernel_init=kernel_init)
    return ma