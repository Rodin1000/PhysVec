using ITensors, ITensorMPS

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_electron(sites, states; linkdims::Int=1)
    # a verifier initialstate for electron system: create an initial MPS state for DMRG
    # sites: a collection of electron site indices
    # states: an array of strings or integers specifying the state at each site (e.g., "Up", "Dn", "Emp", "UpDn")
    # linkdims: the bond dimension of the initial random MPS (default 1)

    if linkdims > 1
        # Create a random MPS with a specific symmetry sector defined by the product state
        psi0 = random_mps(sites, states; linkdims=linkdims)
    else
        # Create a simple product state MPS
        psi0 = MPS(sites, states)
    end

    return psi0
end
