using ITensors, ITensorMPS

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_tJ(sites, states; linkdims::Int=1)
    # a verifier initialstate for tJ system: create an initial MPS state for DMRG
    # sites: vector of site indices (e.g., from siteinds("tJ", N))
    # states: vector of strings defining the state at each site (e.g., "Up", "Dn", "Emp")
    # linkdims: bond dimension for random MPS; if 1, returns a product state (default 1)

    if linkdims > 1
        # Create a random MPS in the quantum number sector defined by states
        psi0 = random_mps(sites, states; linkdims=linkdims)
    else
        # Create a simple product state MPS
        psi0 = MPS(sites, states)
    end

    return psi0
end
