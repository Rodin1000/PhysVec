import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_tVMC(hi, *, observable_type="magnetization", site_index=0):
    # a verifier observable for spin0_5_tVMC system: defines common observables for spin-1/2 tVMC calculations
    # hi: Hilbert space object defining the quantum system (e.g., nk.hilbert.Spin(0.5, N))
    # observable_type: type of observable to create (e.g., "magnetization", "energy", "custom")
    # site_index: index of the site for local observables (default: 0)
    #
    # Creates and returns an observable operator based on the specified type
    # For magnetization, returns sigma_x on specified site
    # For energy, returns the Hamiltonian operator
    # For custom, returns a zero operator as template
    #
    if observable_type == "magnetization":
        obs = nk.operator.spin.sigmax(hi, site_index)
    elif observable_type == "energy":
        # Energy observable would typically be the Hamiltonian
        # This would require additional parameters in practice
        obs = None  # Placeholder - Hamiltonian would need J, h parameters
    else:  # custom
        # Example of custom observable inheriting from AbstractObservable
        class ZeroOperator(nk.experimental.observable.AbstractObservable):
            @property
            def dtype(self):
                return float
            
            @property
            def hilbert(self):
                return hi
                
        obs = ZeroOperator()
    
    return obs