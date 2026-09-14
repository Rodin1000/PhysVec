using ITensors, ITensorMPS

# hamiltonian-----------------------------------------------------------------------------------------
function hamiltonian_verifier_electron(sites, t::Float64, U::Float64; pbc::Bool=false)
    # a verifier hamiltonian for electron system: Hubbard model Hamiltonian for electron sites
    # sites: the array of site indices with "Electron" site type
    # t: hopping amplitude
    # U: on-site interaction strength
    # pbc: whether to use periodic boundary conditions (default: false)

    N = length(sites)
    os = OpSum()
    
    for i in 1:N
        # On-site interaction U * n_up * n_dn
        os += U, "Nupdn", i
        
        # Hopping terms
        j = i + 1
        if j <= N
            # Up spin hopping
            os += -t, "Cdagup", i, "Cup", j
            os += -t, "Cdagup", j, "Cup", i
            # Down spin hopping
            os += -t, "Cdagdn", i, "Cdn", j
            os += -t, "Cdagdn", j, "Cdn", i
        elseif pbc && N > 2
            # Periodic boundary hopping
            os += -t, "Cdagup", N, "Cup", 1
            os += -t, "Cdagup", 1, "Cup", N
            os += -t, "Cdagdn", N, "Cdn", 1
            os += -t, "Cdagdn", 1, "Cdn", N
        end
    end
    
    H = MPO(os, sites)
    return H
end
