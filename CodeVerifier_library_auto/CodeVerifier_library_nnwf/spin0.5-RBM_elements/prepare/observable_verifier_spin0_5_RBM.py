import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_RBM(hi, *, dtype=complex):
    # a verifier observable for spin0_5_RBM system: defines a custom observable for spin-1/2 RBM system
    # hi: Hilbert space object for spin-1/2 system (e.g., nk.hilbert.Spin)
    # dtype: data type for the observable (default: complex)
    #
    # Create a custom observable by subclassing AbstractObservable
    class CustomObservable(nk.experimental.observable.AbstractObservable):
        @property
        def dtype(self):
            return dtype
        
        @property
        def hilbert(self):
            return hi
            
    # Instantiate the observable with the provided Hilbert space
    obs = CustomObservable()
    
    return obs