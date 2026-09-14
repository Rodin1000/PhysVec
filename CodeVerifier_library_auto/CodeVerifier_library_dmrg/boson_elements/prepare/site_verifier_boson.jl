using ITensors, ITensorMPS

# site-----------------------------------------------------------------------------------------
function site_verifier_boson(N::Int; dim::Int=2, conserve_qns::Bool=false, conserve_number::Bool=conserve_qns, qnname_number::String="Number")
    # a verifier site for boson system: creates bosonic lattice sites using siteinds function
    # N: number of lattice sites (integer)
    # dim: dimension of the qudit index, default is 2
    # conserve_qns: whether to conserve total quantum numbers, default is false
    # conserve_number: whether to conserve total boson number, default follows conserve_qns
    # qnname_number: name for the total boson number quantum number, default is "Number"
    
    sites = siteinds("Boson", N; dim=dim, conserve_qns=conserve_qns, conserve_number=conserve_number, qnname_number=qnname_number)
    
    return sites
end