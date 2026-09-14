# Consolidated imports
import netket as nk
import numpy as np


# === compset_verifier_boson_RBM.py ===

# compset-----------------------------------------------------------------------------------------
def compset_verifier_boson_RBM(hi, ma, *, n_samples: int = 1008, n_chains: int = 16, sweep_size: int = 100, learning_rate: float = 0.01):
    # a verifier compset for boson_RBM system: creates sampler, optimizer, and variational state for VMC
    # hi: Hilbert space object (e.g., nk.hilbert.Fock)
    # ma: variational model / ansatz (e.g., nk.models.RBM)
    # n_samples: total number of samples for MCState (default 1008)
    # n_chains: number of Markov chains for sampler (default 16)
    # sweep_size: number of steps per sweep in sampler (default 100)
    # learning_rate: learning rate for SGD optimizer (default 0.01)
    
    # create sampler with MetropolisLocal for local updates
    sa = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=n_chains, sweep_size=sweep_size)
    
    # create optimizer with SGD
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    # create variational state MCState with sampler and model
    vs = nk.vqs.MCState(sampler=sa, model=ma, n_samples=n_samples)
    
    return sa, op, vs


# === effector_verifier_boson_RBM.py ===

# effector-----------------------------------------------------------------------------------------
def effector_verifier_boson_RBM(ha, op, vs):
    # a verifier effector for boson_RBM system: creates VMC driver for variational Monte Carlo optimization
    # ha: Hamiltonian operator defining the problem (e.g., nk.operator.LocalOperator or nk.operator.Ising)
    # op: optimizer for energy minimization (e.g., nk.optimizer.SGD or nk.optimizer.Sgd)
    # vs: variational state object (e.g., nk.vqs.MCState with RBM model)
    
    driv = nk.driver.VMC(ha, op, variational_state=vs)
    
    return driv


# === hamiltonian_verifier_boson_RBM.py ===

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_boson_RBM(hi, g, *, U: float = 1.0, V: float = 0.0, J: float = 1.0, mu: float = 0.0):
    # a verifier hamiltonian for boson_RBM system: Bose-Hubbard Hamiltonian for bosonic systems
    # hi: Fock Hilbert space object for bosons (e.g., nk.hilbert.Fock)
    # g: graph/lattice object defining the connectivity (e.g., nk.graph.Hypercube or nk.graph.Square)
    # U: on-site interaction strength (default 1.0)
    # V: density-density interaction strength between neighbors (default 0.0)
    # J: hopping amplitude (default 1.0)
    # mu: chemical potential (default 0.0)
    
    ha = nk.operator.BoseHubbard(hilbert=hi, U=U, V=V, J=J, mu=mu, graph=g)
    
    return ha


# === hilbert_verifier_boson_RBM.py ===

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_boson_RBM(Lx: int, Ly: int, n_max: int, *, pbc: bool = True, n_particles: int = None):
    # a verifier hilbert for boson_RBM system: creates a 2D square lattice graph and Fock Hilbert space for bosons
    # Lx: number of sites along x-direction
    # Ly: number of sites along y-direction
    # n_max: maximum occupation number per site (inclusive)
    # pbc: periodic boundary conditions (default True)
    # n_particles: constraint on total number of particles (default None, no constraint)
    
    # Create the 2D square lattice graph
    g = nk.graph.Square(length=Lx, pbc=pbc) if Lx == Ly else nk.graph.Grid(extent=[Lx, Ly], pbc=pbc)
    
    # Create the Fock Hilbert space for bosons
    hi = nk.hilbert.Fock(n_max=n_max, N=g.n_nodes, n_particles=n_particles)
    
    return g, hi


# === observable_verifier_boson_RBM.py ===

# observable-----------------------------------------------------------------------------------------
def observable_verifier_boson_RBM(hi, site: int):
    # a verifier observable for boson_RBM system: defines a local number operator observable on a specific site
    # hi: Hilbert space object (e.g., nk.hilbert.Fock) for the bosonic system
    # site: site index where the number operator is measured
    
    # Create the number operator observable on the specified site
    # For bosonic systems, we use the number operator n = b†b
    obs = nk.operator.boson.number(hi, site)
    
    return obs


# === statemodel_verifier_boson_RBM.py ===

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_boson_RBM(alpha: int = 1, *, use_hidden_bias: bool = True):
    # a verifier statemodel for boson_RBM system: creates an RBM variational ansatz model
    # alpha: feature density parameter, determines number of hidden units as alpha * number of visible units
    # use_hidden_bias: whether to use a bias in the hidden layer (default True)
    
    ma = nk.models.RBM(alpha=alpha, use_hidden_bias=use_hidden_bias)
    
    return ma


# === main function ===
def main():
    """
    Main function that chains all element functions for boson_RBM verification.
    Uses minimal parameters for quick execution.
    """
    # Define minimal parameters for quick verification
    Lx = 2          # Small lattice size in x-direction
    Ly = 2          # Small lattice size in y-direction
    n_max = 2       # Maximum occupation per site
    alpha = 1       # RBM feature density
    site = 0        # Site for observable measurement
    
    # Step 1: Create Hilbert space and graph
    g, hi = hilbert_verifier_boson_RBM(Lx=Lx, Ly=Ly, n_max=n_max, pbc=True, n_particles=None)
    print(f"Created graph with {g.n_nodes} nodes and Hilbert space")
    
    # Step 2: Create the RBM model/ansatz
    ma = statemodel_verifier_boson_RBM(alpha=alpha, use_hidden_bias=True)
    print(f"Created RBM model with alpha={alpha}")
    
    # Step 3: Create the Hamiltonian
    ha = hamiltonian_verifier_boson_RBM(hi=hi, g=g, U=1.0, V=0.0, J=1.0, mu=0.0)
    print("Created Bose-Hubbard Hamiltonian")
    
    # Step 4: Create sampler, optimizer, and variational state
    sa, op, vs = compset_verifier_boson_RBM(hi=hi, ma=ma, n_samples=512, n_chains=8, sweep_size=50, learning_rate=0.01)
    print("Created sampler, optimizer, and variational state")
    
    # Step 5: Create observable (number operator on site 0)
    obs = observable_verifier_boson_RBM(hi=hi, site=site)
    print(f"Created number operator observable on site {site}")
    
    # Step 6: Create VMC driver
    driv = effector_verifier_boson_RBM(ha=ha, op=op, vs=vs)
    print("Created VMC driver")
    
    # Print summary
    print("\n=== Verification Summary ===")
    print(f"Lattice: {Lx}x{Ly} square lattice with {g.n_nodes} sites")
    print(f"Hilbert space dimension: {hi.n_states}")
    print(f"Model: RBM with alpha={alpha}")
    print(f"Hamiltonian: Bose-Hubbard (U=1.0, J=1.0)")
    print("All element functions successfully chained!")
    
    return g, hi, ma, ha, sa, op, vs, obs, driv


if __name__ == "__main__":
    main()
