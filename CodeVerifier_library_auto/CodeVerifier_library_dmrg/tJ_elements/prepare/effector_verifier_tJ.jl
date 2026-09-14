using ITensors, ITensorMPS

# effector-----------------------------------------------------------------------------------------
function effector_verifier_tJ(H::MPO, psi0::MPS; nsweeps::Int=5, maxdim::Vector{Int}=[10, 20, 100], cutoff::Vector{Float64}=[1e-10])
    # a verifier effector for tJ system: ground state calculation using DMRG
    # H: the Hamiltonian MPO for the tJ system
    # psi0: the initial MPS state
    # nsweeps: number of sweeps to perform (default: 5)
    # maxdim: array of bond dimensions for each sweep (default: [10, 20, 100])
    # cutoff: array of truncation error thresholds for each sweep (default: [1e-10])
    
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    
    return energy, psi
end
