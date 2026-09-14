import netket as nk

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_tVMC(ha, vs, op, *, t0=0.0, dt=0.01, integrator=None):
    # a verifier effector for spin0_5_tVMC system: creates a TDVP driver for time evolution
    # ha: Hamiltonian operator (e.g., nk.operator.LocalOperator)
    # vs: variational state (e.g., nk.vqs.MCState)
    # op: optimizer (e.g., nk.optimizer.Adam)
    # t0: initial time for evolution (default: 0.0)
    # dt: time step for evolution (default: 0.01)
    # integrator: integrator for adaptive time stepping (default: None)
    
    # Create TDVP driver for time evolution using time-dependent variational principle
    driv = nk.experimental.TDVP(
        ha,           # Hamiltonian operator
        op,           # Optimizer
        vs,           # Variational state
        t0=t0,        # Initial time
        dt=dt,        # Time step
        integrator=integrator  # Optional integrator for adaptive stepping
    )
    
    return driv