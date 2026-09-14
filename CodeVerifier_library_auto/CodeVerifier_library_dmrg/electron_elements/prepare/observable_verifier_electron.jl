using ITensors, ITensorMPS

# observable-----------------------------------------------------------------------------------------
function observable_verifier_electron(psi::MPS, sites::Vector{<:Index}; ops::Vector{String}=["Ntot", "Sz"])
    # a verifier observable for electron system: measure local expectation values and correlation matrices
    # psi: the MPS state to measure
    # sites: the site indices of the system
    # ops: a list of operator names to measure local expectation values (default: ["Ntot", "Sz"])

    results = Dict()

    # Measure local expectation values for each operator in ops
    for op_name in ops
        results[op_name] = expect(psi, op_name)
    end

    # Measure a standard electron correlation matrix (e.g., Cdagup_i Cup_j)
    results["correlation_matrix_up"] = correlation_matrix(psi, "Cdagup", "Cup")

    return results
end
