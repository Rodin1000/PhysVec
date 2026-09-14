using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_spin0_5(N::Int; conserve_sz::Bool=false, qnname_sz::String="TotalSz")
    # a verifier site for spin0_5 system: creates spin-1/2 lattice sites with optional quantum number conservation
    # N: number of sites in the lattice
    # conserve_sz: whether to conserve total Sz quantum number (default: false)
    # qnname_sz: name for the total Sz quantum number label (default: "TotalSz")
    
    sites = siteinds("S=1/2", N; conserve_sz=conserve_sz, qnname_sz=qnname_sz)
    
    return sites
end