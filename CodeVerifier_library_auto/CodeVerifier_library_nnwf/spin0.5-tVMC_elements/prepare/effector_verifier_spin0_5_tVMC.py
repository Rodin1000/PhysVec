import netket as nk

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
