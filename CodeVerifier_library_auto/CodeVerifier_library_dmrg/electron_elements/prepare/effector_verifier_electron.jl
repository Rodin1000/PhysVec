using ITensors, ITensorMPS

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
