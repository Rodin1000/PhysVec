using ITensors, ITensorMPS

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_spin0_5(sites; linkdims::Int=4)
    # a verifier initialstate for spin0_5 system: creates a random matrix product state for spin-1/2 DMRG calculations
    # sites: array of Index objects with "S=1/2" tags created using siteinds("S=1/2", N)
    # linkdims: initial bond dimension for the random MPS (default value: 4)
    
    psi0 = random_mps(sites; linkdims=linkdims)
    
    return psi0
end