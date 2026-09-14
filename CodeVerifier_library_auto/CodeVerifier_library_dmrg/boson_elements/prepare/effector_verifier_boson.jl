using ITensors, ITensorMPS

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