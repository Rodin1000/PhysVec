import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_FFNN(hi, *, dtype=complex):
    # a verifier observable for spin0_5_FFNN system: defines an observable operator for spin0.5-FFNN system
    # hi: Hilbert space object defining the quantum state space for the spin0.5 system
    # dtype: data type of the observable's output (default: complex)
    
    # Create observable as a dictionary containing common observables
    obs = {
        "energy": nk.operator.spin.sigmax(hi, 0),  # Example observable: sigma_x on first site
        "magnetization": nk.operator.spin.sigmaz(hi)  # Example observable: total magnetization
    }
    
    return obs