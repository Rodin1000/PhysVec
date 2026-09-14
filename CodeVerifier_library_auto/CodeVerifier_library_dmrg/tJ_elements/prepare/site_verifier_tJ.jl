using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_tJ(N::Int; conserve_qns::Bool=false, conserve_sz::Bool=false, conserve_nf::Bool=false)
    # a verifier site for tJ system: create site indices for a tJ lattice with optional quantum number conservation
    # N: number of sites in the lattice
    # conserve_qns: general flag to enable quantum number conservation (default: false)
    # conserve_sz: flag to enable spin-z symmetry conservation (default: false)
    # conserve_nf: flag to enable particle number conservation (default: false)
    
    sites = siteinds("tJ", N; 
                     conserve_qns = conserve_qns, 
                     conserve_sz = conserve_sz, 
                     conserve_nf = conserve_nf)
    
    return sites
end
