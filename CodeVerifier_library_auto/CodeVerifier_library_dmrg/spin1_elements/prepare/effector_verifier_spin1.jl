using ITensors, ITensorMPS

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
