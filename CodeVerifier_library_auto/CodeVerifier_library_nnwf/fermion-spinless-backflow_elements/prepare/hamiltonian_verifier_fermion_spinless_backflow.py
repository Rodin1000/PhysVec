import netket as nk
from netket.operator.fermion import destroy as c, create as cdag, number as nc

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_fermion_spinless_backflow(hi, graph, t, V):
    # a verifier hamiltonian for fermion_spinless_backflow system: constructs tight-binding hopping and density-density interaction terms
    # hi: Hilbert space object for the fermion-spinless-backflow system
    # graph: lattice graph object defining the system geometry
    # t: hopping parameter for tight-binding term
    # V: interaction parameter for density-density term
    #
    # Construct Hamiltonian using fermionic operators
    ha = 0.0
    for i, j in graph.edges():
        # Add hopping term
        ha -= t * (cdag(hi, i) @ c(hi, j) + cdag(hi, j) @ c(hi, i))
        # Add density-density interaction term
        ha += V * nc(hi, i) @ nc(hi, j)
    #
    return ha