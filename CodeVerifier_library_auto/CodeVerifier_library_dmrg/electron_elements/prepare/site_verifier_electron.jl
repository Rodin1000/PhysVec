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
