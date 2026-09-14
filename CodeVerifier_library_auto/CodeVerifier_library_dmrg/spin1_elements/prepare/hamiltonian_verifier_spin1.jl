using ITensors, ITensorMPS

# hamiltonian-----------------------------------------------------------------------------------------
function hamiltonian_verifier_spin1(sites, J, D; pbc::Bool=false)
    # a verifier hamiltonian for spin1 system: defines a spin-1 Heisenberg Hamiltonian with single-ion anisotropy
    # sites: the collection of spin-1 site indices
    # J: the exchange coupling constant
    # D: the single-ion anisotropy constant
    # pbc: whether to use periodic boundary conditions (default: false)

    N = length(sites)
    os = OpSum()

    # Nearest-neighbor Heisenberg interactions
    for j in 1:(N - 1)
        os += J, "Sz", j, "Sz", j + 1
        os += J * 0.5, "S+", j, "S-", j + 1
        os += J * 0.5, "S-", j, "S+", j + 1
    end

    # Periodic boundary conditions
    if pbc && N > 2
        os += J, "Sz", N, "Sz", 1
        os += J * 0.5, "S+", N, "S-", 1
        os += J * 0.5, "S-", N, "S+", 1
    end

    # Single-ion anisotropy
    for j in 1:N
        os += D, "Sz", j, "Sz", j
    end

    H = MPO(os, sites)
    return H
end
