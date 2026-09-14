using ITensors, ITensorMPS

# hamiltonian-----------------------------------------------------------------------------------------
function hamiltonian_verifier_boson(sites, t, U, mu; pbc=false)
    # a verifier hamiltonian for boson system: creates a bosonic Hamiltonian MPO for DMRG calculations
    # sites: the site indices for the boson system, created with siteinds("Boson", N; dim=d)
    # t: hopping amplitude between neighboring sites
    # U: on-site interaction strength
    # mu: chemical potential
    # pbc: whether to use periodic boundary conditions (default: false)
    
    os = OpSum()
    N = length(sites)
    
    # Add on-site terms: U*N*(N-1)/2 - mu*N
    for j in 1:N
        os += U/2, "N", j, "N", j
        os += -U/2, "N", j
        os += -mu, "N", j
    end
    
    # Add hopping terms
    for j in 1:N-1
        os += t, "Adag", j, "A", j+1
        os += t, "A", j, "Adag", j+1
    end
    
    # Add periodic boundary condition term if requested
    if pbc && N > 2
        os += t, "Adag", N, "A", 1
        os += t, "A", N, "Adag", 1
    end
    
    H = MPO(os, sites)
    
    return H
end