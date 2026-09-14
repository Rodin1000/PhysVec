# === effector_verifier_spin0_5.jl ===
using ITensors, ITensorMPS

# effector-----------------------------------------------------------------------------------------
function effector_verifier_spin0_5(sites, H, psi0; nsweeps::Int=5, maxdim::Vector{Int}=[10,20,100,100,200], cutoff::Vector{Float64}=[1E-10])
    # a verifier effector for spin0_5 system: performs DMRG calculation for ground state of spin-1/2 system
    # sites: array of site indices for the spin-1/2 chain
    # H: Hamiltonian of the system as an MPO
    # psi0: initial guess for the ground state as an MPS
    # nsweeps: number of DMRG sweeps (default: 5)
    # maxdim: maximum bond dimension for each sweep (default: [10,20,100,100,200])
    # cutoff: truncation error for discarding small singular values (default: [1E-10])
    
    # Perform DMRG calculation to find the ground state
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    
    return energy, psi
end

# === hamiltonian_verifier_spin0_5.jl ===

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

# === initialstate_verifier_spin0_5.jl ===

# initialstate-----------------------------------------------------------------------------------------
function initialstate_verifier_spin0_5(sites; linkdims::Int=4)
    # a verifier initialstate for spin0_5 system: creates a random matrix product state for spin-1/2 DMRG calculations
    # sites: array of Index objects with "S=1/2" tags created using siteinds("S=1/2", N)
    # linkdims: initial bond dimension for the random MPS (default value: 4)
    
    psi0 = random_mps(sites; linkdims=linkdims)
    
    return psi0
end

# === observable_verifier_spin0_5.jl ===

# observable-----------------------------------------------------------------------------------------
function observable_verifier_spin0_5(psi, sites; nsweeps::Int=10, maxdim::Int=100, cutoff::Float64=1e-12, energy_tol::Float64=1e-6)
    # a verifier observable for spin0_5 system: computes energy and correlation functions for spin0.5 system using DMRG
    # psi: initial MPS state for DMRG algorithm
    # sites: site indices defining the spin0.5 system
    # nsweeps: number of DMRG sweeps to perform (default: 10)
    # maxdim: maximum bond dimension during DMRG (default: 100)
    # cutoff: truncation cutoff for SVD during DMRG (default: 1e-12)
    # energy_tol: tolerance for energy convergence in DMRG observer (default: 1e-6)
    
    # Create DMRG observer to track energy convergence
    observer = DMRGObserver(; energy_tol=energy_tol)
    
    # Define Hamiltonian MPO for spin0.5 system (using Heisenberg model as example)
    H = MPO(sites)
    ampo = AutoMPO()
    for j in 1:length(sites)-1
        ampo += 1.0, "Sz", j, "Sz", j+1
        ampo += 0.5, "S+", j, "S-", j+1
        ampo += 0.5, "S-", j, "S+", j+1
    end
    H = MPO(ampo, sites)
    
    # Run DMRG to compute ground state energy
    energy, psi_final = dmrg(H, psi; nsweeps, maxdim, cutoff, observer)
    
    # Compute Sz-Sz correlation function
    zzcorr = correlation_matrix(psi_final, "Sz", "Sz")
    
    # Compute single-site expectations
    sz_expectations = [expect(psi_final, "Sz")[j] for j in 1:length(sites)]
    
    # Return results as a dictionary
    return Dict(
        "energy" => energy,
        "correlation_matrix" => zzcorr,
        "sz_expectations" => sz_expectations,
        "psi_final" => psi_final
    )
end

# === site_verifier_spin0_5.jl ===

# site-----------------------------------------------------------------------------------------
function site_verifier_spin0_5(N::Int; conserve_sz::Bool=false, qnname_sz::String="TotalSz")
    # a verifier site for spin0_5 system: creates spin-1/2 lattice sites with optional quantum number conservation
    # N: number of sites in the lattice
    # conserve_sz: whether to conserve total Sz quantum number (default: false)
    # qnname_sz: name for the total Sz quantum number label (default: "TotalSz")
    
    sites = siteinds("S=1/2", N; conserve_sz=conserve_sz, qnname_sz=qnname_sz)
    
    return sites
end

function main()
    # Define minimal parameters for quick execution
    N = 4  # Small system size for fast computation
    J = 1.0  # Coupling constant
    periodic = false  # Boundary conditions
    
    # Step 1: Create sites (dependency for all other functions)
    sites = site_verifier_spin0_5(N; conserve_sz=false)
    
    # Step 2: Create Hamiltonian using the sites
    H = hamiltonian_verifier_spin0_5(sites, J; periodic=periodic)
    
    # Step 3: Create initial state using the sites
    psi0 = initialstate_verifier_spin0_5(sites; linkdims=4)
    
    # Step 4: Run DMRG calculation to find ground state
    energy, psi = effector_verifier_spin0_5(sites, H, psi0; nsweeps=3, maxdim=[10,20,50], cutoff=[1E-10])
    
    # Step 5: Compute observables using the final state
    results = observable_verifier_spin0_5(psi, sites; nsweeps=5, maxdim=50, cutoff=1e-10, energy_tol=1e-6)
    
    # Print key results
    println("Ground state energy: ", results["energy"])
    println("Sz expectations: ", results["sz_expectations"])
    
    return results
end

# Entry point for the program
main()

