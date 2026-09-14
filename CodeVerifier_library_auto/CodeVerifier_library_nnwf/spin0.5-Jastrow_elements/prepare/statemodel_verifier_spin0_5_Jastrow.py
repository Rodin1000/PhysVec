import netket as nk
import flax.linen as nn
import jax.numpy as jnp

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_Jastrow(hi, *, kernel_init=None):
    # a verifier statemodel for spin0_5_Jastrow system: creates a Jastrow ansatz model for spin-1/2 systems
    # hi: Hilbert space object defining the spin-1/2 system
    # kernel_init: initializer for the Jastrow kernel parameters (default: normal distribution)
    #
    # Creates a Jastrow model with a symmetric kernel matrix of learnable complex parameters
    # that captures pairwise spin correlations beyond mean-field
    #
    if kernel_init is None:
        kernel_init = nn.initializers.normal()
    
    ma = nk.models.Jastrow(kernel_init=kernel_init)
    
    return ma