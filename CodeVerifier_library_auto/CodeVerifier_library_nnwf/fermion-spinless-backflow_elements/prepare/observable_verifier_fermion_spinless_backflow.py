import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_fermion_spinless_backflow(hi, *, indices=None, observable_type="density"):
    # a verifier observable for fermion_spinless_backflow system: defines fermionic observables for spinless fermion systems
    # hi: Hilbert space object for the fermionic system
    # indices: list of site indices where the observable is defined (default: all sites)
    # observable_type: type of observable to create ("density", "correlation", or "custom") (default: "density")
    
    # Default to all sites if indices not specified
    if indices is None:
        indices = list(range(hi.size))
    
    # Create observable based on type
    if observable_type == "density":
        # Density observable: n_i = c^\dagger_i c_i
        obs = nk.operator.fermion.number(hi, site_index=indices[0]) if len(indices) == 1 else \
              sum(nk.operator.fermion.number(hi, site_index=i) for i in indices) / len(indices)
    
    elif observable_type == "correlation":
        # Correlation observable: <c^\dagger_i c_j> for i != j
        if len(indices) != 2:
            raise ValueError("Correlation observable requires exactly two site indices")
        i, j = indices
        obs = nk.operator.fermion.destroy(hi, site_index=i) @ nk.operator.fermion.create(hi, site_index=j)
    
    elif observable_type == "custom":
        # Custom observable can be extended as needed
        # Initialize as zero operator
        obs = nk.operator.fermion.zero(hi)
    
    else:
        raise ValueError(f"Unknown observable type: {observable_type}")
    
    return obs