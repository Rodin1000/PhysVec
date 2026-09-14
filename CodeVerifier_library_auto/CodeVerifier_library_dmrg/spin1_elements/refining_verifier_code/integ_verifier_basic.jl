using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_spin1(N::Int; conserve_qns::Bool=false, conserve_sz::Bool=false)
    # a verifier site for spin1 system: create spin-1 site indices for DMRG
    # N: number of sites in the lattice
    # conserve_qns: whether to conserve quantum numbers (general)
    # conserve_sz: whether to conserve Sz quantum number

    sites = siteinds("S=1", N; conserve_qns=conserve_qns, conserve_sz=conserve_sz)

    return sites
end

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

# effector-----------------------------------------------------------------------------------------
function effector_verifier_spin1(H::MPO, psi0::MPS; nsweeps::Int=5, maxdim::Vector{Int}=[10, 20, 100], cutoff::Vector{Float64}=[1E-10])
    # a verifier effector for spin1 system: perform ground state DMRG calculation
    # H: the Hamiltonian MPO of the spin-1 system
    # psi0: the initial state MPS
    # nsweeps: number of sweeps to perform (default: 5)
    # maxdim: array of maximum bond dimensions for each sweep (default: [10, 20, 100])
    # cutoff: array of truncation error thresholds for each sweep (default: [1E-10])

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    return energy, psi
end

# observable-----------------------------------------------------------------------------------------
function observable_verifier_spin1(psi::MPS, sites::Vector{Index{Int64}}; ops::Vector{String}=["Sz"])
    # a verifier observable for spin1 system: calculate local expectation values and correlation matrices
    # psi: the MPS state obtained from DMRG
    # sites: the site indices for the spin-1 system
    # ops: a list of operator names to measure local expectation values (default: ["Sz"])

    # Measure local expectation values for each operator in ops
    # expect returns a vector of values for each site
    local_expectations = Dict{String, Vector{Float64}}()
    for op_name in ops
        local_expectations[op_name] = expect(psi, op_name)
    end

    # Measure Sz-Sz correlation matrix as a standard spin-1 observable
    # C[i,j] = <psi| Sz_i * Sz_j |psi>
    sz_correlations = correlation_matrix(psi, "Sz", "Sz")

    # Combine results into a dictionary
    results = Dict(
        "local_expectations" => local_expectations,
        "sz_correlations" => sz_correlations
    )

    return results
end

function main()
    # Parameters
    N = 4
    J = 1.0
    D = 0.1
    
    println("Starting verifier for Spin-1 system...")
    
    # 1. Create sites
    sites = site_verifier_spin1(N)
    
    # 2. Create Hamiltonian
    H = hamiltonian_verifier_spin1(sites, J, D)
    
    # 3. Create initial state
    psi0 = initialstate_verifier_spin1(sites)
    
    # 4. Run DMRG (Effector)
    energy, psi = effector_verifier_spin1(H, psi0)
    println("Ground state energy: ", energy)
    
    # 5. Measure observables
    # Note: sites is Vector{Index{Int64}} in site_verifier_spin1
    results = observable_verifier_spin1(psi, sites)
    
    println("Local Sz expectations: ", results["local_expectations"]["Sz"])
    println("Verification complete.")
end

main()
