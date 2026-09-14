import netket as nk
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_FFNN(*, features_factor: int = 2, use_bias: bool = True, param_dtype: type = np.complex128, kernel_stddev: float = 0.01, bias_stddev: float = 0.01):
    # a verifier statemodel for spin0_5_FFNN system: defines a feed-forward neural network model for spin 1/2 systems using flax.linen Module
    # features_factor: multiplier for the number of features in the Dense layer, calculated as features_factor * x.shape[-1] (default: 2)
    # use_bias: whether to use bias in the Dense layer (default: True)
    # param_dtype: data type for parameters (default: np.complex128)
    # kernel_stddev: standard deviation for kernel initialization (default: 0.01)
    # bias_stddev: standard deviation for bias initialization (default: 0.01)
    
    class FFNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            # Apply Dense layer with features = features_factor * x.shape[-1]
            x = nn.Dense(
                features=features_factor * x.shape[-1],
                use_bias=use_bias,
                param_dtype=param_dtype,
                kernel_init=nn.initializers.normal(kernel_stddev),
                bias_init=nn.initializers.normal(bias_stddev)
            )(x)
            # Apply log_cosh activation
            x = jnp.log(jnp.cosh(x))
            # Sum over the last axis
            return jnp.sum(x, axis=-1)
    
    # Instantiate and return the model object
    ma = FFNN()
    return ma