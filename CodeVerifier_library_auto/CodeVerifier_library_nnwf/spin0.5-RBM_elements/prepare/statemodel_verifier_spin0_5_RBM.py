import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_RBM(*, alpha: float = 1.0):
    # a verifier statemodel for spin0_5_RBM system: creates an RBM model for spin-1/2 systems
    # alpha: scaling factor for number of hidden units (hidden_units = alpha * visible_units), controls model capacity (default: 1.0)
    #
    # Creates a Restricted Boltzmann Machine (RBM) model with specified alpha parameter
    # The model is used with variational states like MCState for quantum simulations
    #
    ma = nk.models.RBM(alpha=alpha)
    return ma