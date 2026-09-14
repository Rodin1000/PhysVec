using ITensors, ITensorMPS

# observable-----------------------------------------------------------------------------------------
function observable_verifier_spin1(psi::MPS, sites::Vector{Index{Int64}}; ops::Vector{String}=["Sz"])
    # a verifier observable for spin1 system: calculate local expectation values and correlation matrices
    # psi: the MPS state obtained from DMRG
    # sites: the site indices for the spin-1 system
    # ops: a list of operator names to measure local expectation values (default: ["Sz"])

    # Measure local expectation values for each operator in ops
    # expect returns a vector of values for each site
    local_expectations = Dict{String, Vector{Float64}}()
    for op_name in ops
        local_expectations[op_name] = expect(psi, op_name)
    end

    # Measure Sz-Sz correlation matrix as a standard spin-1 observable
    # C[i,j] = <psi| Sz_i * Sz_j |psi>
    sz_correlations = correlation_matrix(psi, "Sz", "Sz")

    # Combine results into a dictionary
    results = Dict(
        "local_expectations" => local_expectations,
        "sz_correlations" => sz_correlations
    )

    return results
end
