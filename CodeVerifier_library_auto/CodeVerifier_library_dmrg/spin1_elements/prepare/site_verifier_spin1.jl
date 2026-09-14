using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_spin1(N::Int; conserve_qns::Bool=false, conserve_sz::Bool=false)
    # a verifier site for spin1 system: create spin-1 site indices for DMRG
    # N: number of sites in the lattice
    # conserve_qns: whether to conserve quantum numbers (general)
    # conserve_sz: whether to conserve Sz quantum number

    sites = siteinds("S=1", N; conserve_qns=conserve_qns, conserve_sz=conserve_sz)

    return sites
end
