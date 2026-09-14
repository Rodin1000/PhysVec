import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_tVMC(hi):
    # a verifier statemodel for spin0_5_tVMC system: a Jastrow ansatz model for spin-1/2 systems
    # hi: the Hilbert space object (e.g., nk.hilbert.Spin)
    #
    ma = nk.models.Jastrow()
    return ma
