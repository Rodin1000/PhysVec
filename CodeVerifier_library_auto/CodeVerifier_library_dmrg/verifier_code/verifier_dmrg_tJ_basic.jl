using ITensors, ITensorMPS

# effector-----------------------------------------------------------------------------------------
function effector_verifier_tJ(H::MPO, psi0::MPS; nsweeps::Int=5, maxdim::Vector{Int}=[10, 20, 100], cutoff::Vector{Float64}=[1e-10], noise::Vector{Float64}=[1e-6, 1e-7, 1e-8, 0.0])
    # a verifier effector for tJ system: ground state calculation using DMRG
    # H: the Hamiltonian MPO for the tJ system
    # psi0: the initial MPS state
    # nsweeps: number of sweeps to perform (default: 5)
    # maxdim: array of bond dimensions for each sweep (default: [10, 20, 100])
    # cutoff: array of truncation error thresholds for each sweep (default: [1e-10])
    # noise: array of noise values to avoid local minima (default: [1e-6, 1e-7, 1e-8, 0.0])
    
    # Use DMRGObserver to monitor convergence and stop early if energy_tol is reached
    observer = DMRGObserver(; energy_tol=1e-8)
    
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise, observer=observer)
    
    return energy, psi
end


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


# observable-----------------------------------------------------------------------------------------
function observable_verifier_tJ(psi::MPS, sites; ops::Vector{String}=["Sz", "Ntot"])
    # a verifier observable for tJ system: calculate expectation values and correlation matrices
    # psi: the MPS state to measure
    # sites: the site indices of the system
    # ops: a list of local operator names to measure expectation values (default: ["Sz", "Ntot"])

    results = Dict()

    # Measure expectation values for each operator in the list
    for op_name in ops
        results[op_name] = expect(psi, op_name)
    end

    # Measure spin-spin correlation matrix
    results["SzSz_correlation"] = correlation_matrix(psi, "Sz", "Sz")

    # Measure electron-electron correlation matrix (total density)
    results["NtotNtot_correlation"] = correlation_matrix(psi, "Ntot", "Ntot")

    return results
end


# site-----------------------------------------------------------------------------------------
function site_verifier_tJ(N::Int; conserve_qns::Bool=false, conserve_sz::Bool=false, conserve_nf::Bool=false)
    # a verifier site for tJ system: create site indices for a tJ lattice with optional quantum number conservation
    # N: number of sites in the lattice
    # conserve_qns: general flag to enable quantum number conservation (default: false)
    # conserve_sz: flag to enable spin-z symmetry conservation (default: false)
    # conserve_nf: flag to enable particle number conservation (default: false)
    
    sites = siteinds("tJ", N; 
                     conserve_qns = conserve_qns, 
                     conserve_sz = conserve_sz, 
                     conserve_nf = conserve_nf)
    
    return sites
end

function main()
    # Parameters
    N = 4
    t = 1.0
    J = 0.5
    
    # 1. Create sites
    sites = site_verifier_tJ(N; conserve_qns=false)
    
    # 2. Create Hamiltonian
    H = hamiltonian_verifier_tJ(sites, t, J)
    
    # 3. Create initial state (e.g., half-filling, alternating spins)
    states = ["Up", "Dn", "Up", "Dn"]
    psi0 = initialstate_verifier_tJ(sites, states)
    
    # 4. Run DMRG (Effector)
    # Using a noise schedule to help convergence and avoid local minima
    nsweeps = 3
    maxdims = [10, 20, 40]
    cutoffs = [1e-8]
    noises = [1e-6, 1e-8, 0.0]
    
    energy, psi = effector_verifier_tJ(H, psi0; nsweeps=nsweeps, maxdim=maxdims, cutoff=cutoffs, noise=noises)
    
    println("Ground State Energy: ", energy)
    
    # 5. Measure observables
    results = observable_verifier_tJ(psi, sites)
    
    println("Expectation values (Sz): ", results["Sz"])
    println("Expectation values (Ntot): ", results["Ntot"])
end

main()
