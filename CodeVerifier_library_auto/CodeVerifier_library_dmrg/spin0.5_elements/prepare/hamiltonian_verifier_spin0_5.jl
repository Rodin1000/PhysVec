using ITensors, ITensorMPS

# hamiltonian-----------------------------------------------------------------------------------------
function hamiltonian_verifier_spin0_5(sites, J::Float64=1.0; periodic::Bool=false)
    # a verifier hamiltonian for spin0_5 system: Heisenberg model Hamiltonian for spin-1/2 systems
    # sites: array of site indices created with siteinds("S=1/2", N)
    # J: coupling constant for Heisenberg interactions (default value: 1.0)
    # periodic: whether to use periodic boundary conditions (default value: false)
    
    # Create OpSum to collect Hamiltonian terms
    os = OpSum()
    N = length(sites)
    
    # Add Heisenberg terms for nearest neighbor interactions
    for j in 1:(N-1)
        # Sz Sz interaction
        os += "Sz", j, "Sz", j+1
        # S+ S- and S- S+ interactions (factor of 0.5 each)
        os += 0.5, "S+", j, "S-", j+1
        os += 0.5, "S-", j, "S+", j+1
    end
    
    # Add periodic boundary term if requested
    if periodic
        os += "Sz", N, "Sz", 1
        os += 0.5, "S+", N, "S-", 1
        os += 0.5, "S-", N, "S+", 1
    end
    
    # Convert OpSum to MPO
    H = MPO(os, sites)
    
    return H
end