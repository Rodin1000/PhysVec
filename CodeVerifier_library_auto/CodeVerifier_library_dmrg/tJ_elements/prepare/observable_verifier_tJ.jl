using ITensors, ITensorMPS

# observable-----------------------------------------------------------------------------------------
function observable_verifier_tJ(psi::MPS, sites; ops::Vector{String}=["Sz", "Ntot"])
    # a verifier observable for tJ system: calculate expectation values and correlation matrices
    # psi: the MPS state to measure
    # sites: the site indices of the system
    # ops: a list of local operator names to measure expectation values (default: ["Sz", "Ntot"])

    results = Dict()

    # Measure expectation values for each operator in the list
    for op_name in ops
        results[op_name] = expect(psi, op_name)
    end

    # Measure spin-spin correlation matrix
    results["SzSz_correlation"] = correlation_matrix(psi, "Sz", "Sz")

    # Measure electron-electron correlation matrix (total density)
    results["NtotNtot_correlation"] = correlation_matrix(psi, "Ntot", "Ntot")

    return results
end
