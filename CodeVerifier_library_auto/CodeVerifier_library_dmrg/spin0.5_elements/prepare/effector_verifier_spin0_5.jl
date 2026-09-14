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