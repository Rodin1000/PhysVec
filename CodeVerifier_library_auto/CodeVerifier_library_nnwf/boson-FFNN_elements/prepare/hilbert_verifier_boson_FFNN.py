import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_boson_FFNN(N: int, n_max: int, *, n_particles: int = None):
    # a verifier hilbert for boson_FFNN system: defines a Fock (boson) Hilbert space with specified sites and max occupation
    # N: number of bosonic modes/sites in the Hilbert space
    # n_max: maximum occupation number per site (inclusive), determines local dimension as (n_max + 1)
    # n_particles: optional constraint for total number of particles (default None means no constraint)
    
    hi = nk.hilbert.Fock(n_max=n_max, N=N, n_particles=n_particles)
    
    return hi
