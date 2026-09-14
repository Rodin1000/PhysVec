using ITensors, ITensorMPS

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
    observer = DMRGObserver(["energy"], energy_tol=energy_tol)
    
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
    sz_expectations = [expect(psi_final, "Sz", j) for j in 1:length(sites)]
    
    # Return results as a dictionary
    return Dict(
        "energy" => energy,
        "correlation_matrix" => zzcorr,
        "sz_expectations" => sz_expectations,
        "psi_final" => psi_final
    )
end