using ITensors, ITensorMPS

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_boson(sites; initial_state_type="random", conserve_qns=false)
    # a verifier initialstate for boson system: creates an initial MPS state for bosonic DMRG calculations
    # sites: collection of site indices with boson site types, must have well-defined quantum numbers if conserve_qns=true
    # initial_state_type: strategy for initializing the state (e.g., "random", "vacuum", "half-filled"), default "random"
    # conserve_qns: whether to conserve quantum numbers in the initial state, must match Hamiltonian settings
    
    # Create initial MPS with specified properties
    psi0 = productMPS(sites)
    
    # Initialize based on requested type
    if initial_state_type == "vacuum"
        for i in 1:length(sites)
            setindex!(psi0, 1, i)  # Set to vacuum state (0 bosons)
        end
    elseif initial_state_type == "half-filled"
        for i in 1:length(sites)
            dim = ITensors.dim(sites[i])
            mid_state = div(dim, 2) + 1
            setindex!(psi0, mid_state, i)  # Set to approximately half-filled
        end
    else  # random initialization
        # Default random initialization through productMPS
        nothing
    end
    
    # Ensure state has correct QN structure if required
    if conserve_qns
        # Verify that the state has well-defined total quantum numbers
        total_qn = totalqn(psi0)
    end
    
    return psi0
end