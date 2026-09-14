using ITensors, ITensorMPS

# hamiltonian-----------------------------------------------------------------------------------------
function hamiltonian_verifier_tJ(sites, t::Float64, J::Float64; pbc::Bool=false)
    # a verifier hamiltonian for tJ system: defines the t-J model Hamiltonian using OpSum and MPO
    # sites: the array of site indices (typically "tJ" or "Electron" site types)
    # t: hopping amplitude
    # J: Heisenberg exchange coupling
    # pbc: boolean flag for periodic boundary conditions (default: false)

    N = length(sites)
    os = OpSum()

    # Hopping terms (t) and Heisenberg exchange terms (J)
    for j in 1:(N - 1)
        # Hopping terms
        os += -t, "Cdagup", j, "Cup", j + 1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
        os += -t, "Cdagdn", j + 1, "Cdn", j

        # Heisenberg exchange terms
        os += J, "Sz", j, "Sz", j + 1
        os += 0.5 * J, "S+", j, "S-", j + 1
        os += 0.5 * J, "S-", j, "S+", j + 1
    end

    if pbc && N > 2
        # Periodic boundary conditions
        os += -t, "Cdagup", N, "Cup", 1
        os += -t, "Cdagup", 1, "Cup", N
        os += -t, "Cdagdn", N, "Cdn", 1
        os += -t, "Cdagdn", 1, "Cdn", N

        os += J, "Sz", N, "Sz", 1
        os += 0.5 * J, "S+", N, "S-", 1
        os += 0.5 * J, "S-", N, "S+", 1
    end

    return MPO(os, sites)
end
