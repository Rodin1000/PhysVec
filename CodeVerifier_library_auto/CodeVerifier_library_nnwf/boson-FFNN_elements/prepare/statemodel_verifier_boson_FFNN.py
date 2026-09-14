import netket as nk
import netket.nn as nknn
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

# statemodel-----------------------------------------------------------------------------------------
class FFNN(nn.Module):
    # feed-forward neural network model for boson systems
    # alpha: feature density (hidden layer size = alpha * input size)
    alpha: int = 1
    
    @nn.compact
    def __call__(self, x):
        # x: input configuration with shape (n_samples, N)
        # apply dense layer with complex parameters
        x = nn.Dense(features=self.alpha * x.shape[-1], 
                     use_bias=True, 
                     param_dtype=np.complex128,
                     kernel_init=nn.initializers.normal(stddev=0.01))(x)
        # apply activation function
        x = nknn.log_cosh(x)
        # sum over features to get scalar log-amplitude
        return jnp.sum(x, axis=-1)

def statemodel_verifier_boson_FFNN(alpha: int = 1):
    # a verifier statemodel for boson_FFNN system: creates a feed-forward neural network model
    # alpha: feature density determining hidden layer size as alpha * input_size (default 1)
    #
    # create FFNN model instance with specified alpha
    ma = FFNN(alpha=alpha)
    #
    return ma
