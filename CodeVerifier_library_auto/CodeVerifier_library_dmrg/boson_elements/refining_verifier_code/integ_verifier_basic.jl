# === effector_verifier_boson.jl ===
using ITensors, ITensorMPS

# === site_verifier_boson.jl ===
# site-----------------------------------------------------------------------------------------
function site_verifier_boson(N::Int; dim::Int=2, conserve_qns::Bool=false, conserve_number::Bool=conserve_qns, qnname_number::String="Number")
    # a verifier site for boson system: creates bosonic lattice sites using siteinds function
    # N: number of lattice sites (integer)
    # dim: dimension of the qudit index, default is 2
    # conserve_qns: whether to conserve total quantum numbers, default is false
    # conserve_number: whether to conserve total boson number, default follows conserve_qns
    # qnname_number: name for the total boson number quantum number, default is "Number"
    
    sites = siteinds("Boson", N; dim=dim, conserve_qns=conserve_qns, conserve_number=conserve_number, qnname_number=qnname_number)
    
    return sites
end

# === hamiltonian_verifier_boson.jl ===
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

# === initialstate_verifier_boson.jl ===
# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_boson(sites; initial_state_type="random", conserve_qns=false)
    # a verifier initialstate for boson system: creates an initial MPS state for bosonic DMRG calculations
    # sites: collection of site indices with boson site types, must have well-defined quantum numbers if conserve_qns=true
    # initial_state_type: strategy for initializing the state (e.g., "random", "vacuum", "half-filled"), default "random"
    # conserve_qns: whether to conserve quantum numbers in the initial state, must match Hamiltonian settings
    
    # Create initial MPS with specified properties
    psi0 = randomMPS(sites)
    
    # Initialize based on requested type
    if initial_state_type == "vacuum"
        for i in 1:length(sites)
            setindex!(psi0, 1, i)  # Set to vacuum state (0 bosons)
        end
    elseif initial_state_type == "half-filled"
        for i in 1:length(sites)
            dim = ITensors.dim(sites[i])
            mid_state = div(dim, 2) + 1
            setindex!(psi0, mid_state, i)  # Set to approximately half-filled
        end
    else  # random initialization
        # Default random initialization through productMPS
        nothing
    end
    
    # Ensure state has correct QN structure if required
    if conserve_qns
        # Verify that the state has well-defined total quantum numbers
        total_qn = totalqn(psi0)
    end
    
    return psi0
end

# === effector_verifier_boson.jl ===
# effector-----------------------------------------------------------------------------------------
function effector_verifier_boson(sites, H, psi0; nsweeps::Int=10, maxdim::Vector{Int}=[10,20,100,100,200], cutoff::Float64=1E-10, weight::Float64=1.0)
    # a verifier effector for boson system: perform DMRG calculation for ground or excited states
    # sites: bosonic site indices created with siteinds("Boson", N)
    # H: Hamiltonian MPO for the bosonic system
    # psi0: initial state MPS for DMRG
    # nsweeps: number of DMRG sweeps (default: 10)
    # maxdim: maximum bond dimensions for each sweep (default: [10,20,100,100,200])
    # cutoff: truncation error for SVD during sweeps (default: 1E-10)
    # weight: penalty weight for excited state calculations (default: 1.0)
    
    # Perform DMRG calculation for ground state
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    
    # Return the ground state energy and MPS
    return energy, psi
end

# === observable_verifier_boson.jl ===
# observable-----------------------------------------------------------------------------------------
function observable_verifier_boson(sites, psi, H; observable_type="energy")
    # a verifier observable for boson system: calculates specified observables for bosonic DMRG simulations
    # sites: site indices for the boson system (array of Index objects)
    # psi: matrix product state (MPS) representing the quantum state
    # H: Hamiltonian MPO for the system
    # observable_type: type of observable to calculate ("energy", "correlation", "local")
    
    if observable_type == "energy"
        # Calculate energy expectation value <psi|H|psi>
        energy = inner(psi', H, psi)
        return energy
    elseif observable_type == "correlation"
        # Calculate correlation matrix <psi|b_i^\dagger b_j|psi>
        corr = correlation_matrix(psi, "Adag", "A")
        return corr
    elseif observable_type == "local"
        # Calculate local boson number <psi|n_i|psi> for each site
        local_obs = [
            expect(psi, "N"; sites=i) for i in 1:length(sites)
        ]
        return local_obs
    else
        error("Unsupported observable type: $observable_type")
    end
end

function main()
    # Define minimal parameters for quick verification
    N = 4  # Small system size
    dim = 2  # Small dimension
    t = 1.0  # Hopping amplitude
    U = 2.0  # On-site interaction
    mu = 0.5  # Chemical potential
    
    # Create sites
    sites = site_verifier_boson(N; dim=dim, conserve_qns=false)
    
    # Create Hamiltonian
    H = hamiltonian_verifier_boson(sites, t, U, mu; pbc=false)
    
    # Create initial state
    psi0 = initialstate_verifier_boson(sites; initial_state_type="random", conserve_qns=false)
    
    # Perform DMRG calculation
    energy, psi = effector_verifier_boson(sites, H, psi0)
    
    # Calculate observables
    energy_obs = observable_verifier_boson(sites, psi, H; observable_type="energy")
    corr_obs = observable_verifier_boson(sites, psi, H; observable_type="correlation")
    local_obs = observable_verifier_boson(sites, psi, H; observable_type="local")
    
    # Print results
    println("Ground state energy: ", energy)
    println("Energy from observable: ", energy_obs)
    println("Correlation matrix: ", corr_obs)
    println("Local observables: ", local_obs)
    
    return energy, psi, energy_obs, corr_obs, local_obs
end

main()

