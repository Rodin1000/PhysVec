using ITensors, ITensorMPS

# observable-----------------------------------------------------------------------------------------
function observable_verifier_boson(sites, psi, H; observable_type="energy")
    # a verifier observable for boson system: calculates specified observables for bosonic DMRG simulations
    # sites: site indices for the boson system (array of Index objects)
    # psi: matrix product state (MPS) representing the quantum state
    # H: Hamiltonian MPO for the system
    # observable_type: type of observable to calculate ("energy", "correlation", "local")
    
    if observable_type == "energy"
        # Calculate energy expectation value <psi|H|psi>
        energy = inner(psi, H, psi)
        return energy
    elseif observable_type == "correlation"
        # Calculate correlation matrix <psi|b_i^\dagger b_j|psi>
        corr = correlation_matrix(psi, "Bd", "B")
        return corr
    elseif observable_type == "local"
        # Calculate local boson number <psi|n_i|psi> for each site
        local_obs = [
            expect(psi, "N"; sites=i) for i in 1:length(sites)
        ]
        return local_obs
    else
        error("Unsupported observable type: $observable_type")
    end
end