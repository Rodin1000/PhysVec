using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_electron(N::Int; conserve_qns::Bool=false, conserve_sz::Bool=false, conserve_nf::Bool=false, conserve_nfparity::Bool=false)
    # a verifier site for electron system: create electron site indices with optional quantum number conservation
    # N: number of sites in the lattice
    # conserve_qns: whether to conserve both total electron number and total spin (default: false)
    # conserve_sz: whether to conserve total spin Sz (default: false)
    # conserve_nf: whether to conserve total electron number Nf (default: false)
    # conserve_nfparity: whether to conserve electron number parity (default: false)

    sites = siteinds("Electron", N; 
                     conserve_qns=conserve_qns, 
                     conserve_sz=conserve_sz, 
                     conserve_nf=conserve_nf, 
                     conserve_nfparity=conserve_nfparity)

    return sites
end

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_electron(sites, states; linkdims::Int=1)
    # a verifier initialstate for electron system: create an initial MPS state for DMRG
    # sites: a collection of electron site indices
    # states: an array of strings or integers specifying the state at each site (e.g., "Up", "Dn", "Emp", "UpDn")
    # linkdims: the bond dimension of the initial random MPS (default 1)

    if linkdims > 1
        # Create a random MPS with a specific symmetry sector defined by the product state
        psi0 = random_mps(sites, states; linkdims=linkdims)
    else
        # Create a simple product state MPS
        psi0 = MPS(sites, states)
    end

    return psi0
end

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

# effector-----------------------------------------------------------------------------------------
function effector_verifier_electron(H, psi0; nsweeps::Int=5, maxdim::Vector{Int}=[10, 20, 100], cutoff::Vector{Float64}=[1e-10])
    # a verifier effector for electron system: run dmrg to find the ground state energy and wavefunction
    # H: the Hamiltonian MPO
    # psi0: the initial state MPS
    # nsweeps: number of sweeps to perform (default 5)
    # maxdim: maximum bond dimension for each sweep (default [10, 20, 100])
    # cutoff: truncation error threshold for each sweep (default [1e-10])

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)

    return energy, psi
end

# observable-----------------------------------------------------------------------------------------
function observable_verifier_electron(psi::MPS, sites::Vector{<:Index}; ops::Vector{String}=["Ntot", "Sz"])
    # a verifier observable for electron system: measure local expectation values and correlation matrices
    # psi: the MPS state to measure
    # sites: the site indices of the system
    # ops: a list of operator names to measure local expectation values (default: ["Ntot", "Sz"])

    results = Dict()

    # Measure local expectation values for each operator in ops
    for op_name in ops
        results[op_name] = expect(psi, op_name)
    end

    # Measure a standard electron correlation matrix (e.g., Cdagup_i Cup_j)
    results["correlation_matrix_up"] = correlation_matrix(psi, "Cdagup", "Cup")

    return results
end

function main()
    # Parameters
    N = 4
    t = 1.0
    U = 4.0
    
    # 1. Create sites
    sites = site_verifier_electron(N; conserve_qns=true)
    
    # 2. Create initial state (Half-filling: 2 up, 2 down)
    states = ["Up", "Dn", "Up", "Dn"]
    psi0 = initialstate_verifier_electron(sites, states)
    
    # 3. Create Hamiltonian
    H = hamiltonian_verifier_electron(sites, t, U)
    
    # 4. Run DMRG
    energy, psi = effector_verifier_electron(H, psi0; nsweeps=3, maxdim=[10, 20], cutoff=[1e-8])
    
    println("Ground State Energy: ", energy)
    
    # 5. Measure observables
    results = observable_verifier_electron(psi, sites)
    
    println("Local Occupancy (Ntot): ", results["Ntot"])
    println("Local Sz: ", results["Sz"])
end

main()
