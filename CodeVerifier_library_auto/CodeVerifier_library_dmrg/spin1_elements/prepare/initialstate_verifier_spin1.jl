using ITensors, ITensorMPS

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_spin1(sites; state_type::String="product", linkdims::Int=1)
    # a verifier initialstate for spin1 system: creates an initial MPS state for DMRG
    # sites: the site indices of the spin-1 system
    # state_type: type of initial state to create ("product" or "random") (default "product")
    # linkdims: bond dimension for random MPS (default 1)

    N = length(sites)
    
    # Define a default state configuration (alternating Up and Dn)
    # For S=1, valid strings are "Up", "Z0", "Dn"
    state_config = [isodd(n) ? "Up" : "Dn" for n in 1:N]

    if state_type == "product"
        psi0 = MPS(sites, state_config)
    elseif state_type == "random"
        # For QN conservation, random_mps needs a state configuration to define the sector
        if hasqns(sites[1])
            psi0 = random_mps(sites, state_config; linkdims=linkdims)
        else
            psi0 = random_mps(sites; linkdims=linkdims)
        end
    else
        error("Unknown state_type: $state_type")
    end

    return psi0
end
