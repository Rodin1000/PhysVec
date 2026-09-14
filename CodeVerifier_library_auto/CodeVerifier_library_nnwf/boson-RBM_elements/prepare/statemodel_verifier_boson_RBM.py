import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_boson_RBM(alpha: int = 1, *, use_hidden_bias: bool = True):
    # a verifier statemodel for boson_RBM system: creates an RBM variational ansatz model
    # alpha: feature density parameter, determines number of hidden units as alpha * number of visible units
    # use_hidden_bias: whether to use a bias in the hidden layer (default True)
    
    ma = nk.models.RBM(alpha=alpha, use_hidden_bias=use_hidden_bias)
    
    return ma
