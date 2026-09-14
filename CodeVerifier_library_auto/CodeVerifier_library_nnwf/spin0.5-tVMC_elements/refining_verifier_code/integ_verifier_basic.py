import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_tVMC(g, *, s: float = 0.5, total_sz: float = None):
    # a verifier hilbert for spin0_5_tVMC system: defines a spin-1/2 Hilbert space on a given graph
    # g: the graph or lattice object defining the number of sites
    # s: the spin quantum number (default 0.5 for spin-1/2)
    # total_sz: optional total magnetization constraint (sum of Sz)
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_tVMC(hi):
    # a verifier statemodel for spin0_5_tVMC system: a Jastrow ansatz model for spin-1/2 systems
    # hi: the Hilbert space object (e.g., nk.hilbert.Spin)
    #
    ma = nk.models.Jastrow()
    return ma

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin, ma, *, n_samples: int = 1024, learning_rate: float = 0.01):
    # a verifier compset for spin0_5_tVMC system: defines the sampler, optimizer, and variational state
    # hi: the Hilbert space object for the spin 0.5 system
    # ma: the variational model or ansatz (e.g., RBM)
    # n_samples: number of Monte Carlo samples to use in the variational state (default: 1024)
    # learning_rate: the learning rate for the optimizer (default: 0.01)

    # Define the sampler
    sa = nk.sampler.MetropolisLocal(hi)

    # Define the optimizer
    op = nk.optimizer.Sgd(learning_rate=learning_rate)

    # Define the variational quantum state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)

    return sa, op, vs

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin, g: nk.graph.Graph, J: float, h: float):
    # a verifier hamiltonian for spin0_5_tVMC system: transverse field Ising model on a graph
    # hi: the Hilbert space object for spin-1/2 particles
    # g: the graph or lattice object defining the connectivity
    # J: the interaction strength between adjacent spins
    # h: the transverse field strength
    #
    ha = nk.operator.Ising(hilbert=hi, graph=g, J=J, h=h)
    #
    return ha

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_tVMC(ha, vs, integrator, *, t0: float = 0.0, propagation_type: str = 'real'):
    # a verifier effector for spin0_5_tVMC system: initialize a TDVP driver for time evolution
    # ha: the Hamiltonian operator (generator of time evolution)
    # vs: the variational state (e.g., nk.vqs.MCState)
    # integrator: the ODE solver/integrator (e.g., nk.experimental.dynamics.RK45)
    # t0: the initial time (default: 0.0)
    # propagation_type: the type of time evolution, 'real' or 'imaginary' (default: 'real')
    
    driv = nk.experimental.TDVP(
        ha,
        vs,
        integrator,
        t0=t0,
        propagation_type=propagation_type
    )
    
    return driv

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_tVMC(hi: nk.hilbert.Spin):
    # a verifier observable for spin0_5_tVMC system: define a total magnetization observable
    # hi: the spin-1/2 Hilbert space object
    #
    # The observable is defined as the sum of sigma_z operators over all sites
    n_sites = hi.size
    obs = sum([nk.operator.spin.sigmaz(hi, i) for i in range(n_sites)])
    
    return obs

def main():
    # 1. Setup Graph (Small system size for verification)
    L = 4
    g = nk.graph.Chain(length=L)

    # 2. Define Hilbert Space
    hi = hilbert_verifier_spin0_5_tVMC(g)

    # 3. Define Variational Model
    ma = statemodel_verifier_spin0_5_tVMC(hi)

    # 4. Setup Compset (Sampler, Optimizer, MCState)
    sa, op, vs = compset_verifier_spin0_5_tVMC(hi, ma, n_samples=512)

    # 5. Define Hamiltonian
    ha = hamiltonian_verifier_spin0_5_tVMC(hi, g, J=1.0, h=1.0)

    # 6. Define Observable
    obs = observable_verifier_spin0_5_tVMC(hi)

    # 7. Setup Effector (TDVP Driver)
    # Using a simple RK4 integrator for verification
    try:
        import netket.experimental.dynamics as nkdynamics
        integrator = nkdynamics.RK4(dt=0.01)
    except (ImportError, AttributeError):
        # Fallback to a standard integrator if the experimental one is not found
        # In modern NetKet, we often use solvers from the 'netket.optimizer.solver' 
        # but for TDVP we usually need a dynamics integrator.
        # Let's try to use a simple Euler if available or just a fixed dt.
        try:
            from netket.experimental.dynamics import Euler
            integrator = Euler(dt=0.01)
        except:
            # If all else fails, we might need to check the specific NetKet version
            # For this verifier, we'll try to use a common one.
            raise ImportError("Could not find a suitable NetKet integrator for TDVP.")

    driv = effector_verifier_spin0_5_tVMC(ha, vs, integrator)

    # 8. Run a single step to verify integration
    print("Starting verification run...")
    driv.run(T=0.01, out=None)
    print("Verification successful: TDVP step completed.")

if __name__ == "__main__":
    main()
