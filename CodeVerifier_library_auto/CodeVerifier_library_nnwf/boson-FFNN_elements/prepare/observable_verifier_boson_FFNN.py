import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_boson_FFNN(hi, sites: list = None, *, return_dict: bool = False):
    # a verifier observable for boson_FFNN system: defines boson number observables for VMC calculation
    # hi: Hilbert space object (nk.hilbert.Fock) for the boson system
    # sites: list of site indices to measure number operator (default: all sites)
    # return_dict: if True, return dictionary of observables; if False, return total number operator (default: False)
    
    # Get number of sites from Hilbert space
    N = hi.size
    
    # Default to all sites if not specified
    if sites is None:
        sites = list(range(N))
    
    if return_dict:
        # Create dictionary of observables for individual site occupations
        obs = {}
        for i in sites:
            obs[f'n_{i}'] = nk.operator.boson.number(hi, i)
        # Add total particle number
        obs['n_total'] = sum([nk.operator.boson.number(hi, i) for i in sites])
    else:
        # Return total number operator as single observable
        obs = sum([nk.operator.boson.number(hi, i) for i in sites])
    
    return obs
