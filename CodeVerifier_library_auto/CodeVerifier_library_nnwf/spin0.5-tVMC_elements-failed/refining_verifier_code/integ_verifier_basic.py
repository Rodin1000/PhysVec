# === compset_verifier_spin0_5_tVMC.py ===
import netket as nk
from netket.experimental import TDVP
# from netket.dynamics import RK4
import flax.linen as nn

# === hilbert_verifier_spin0_5_tVMC.py ===
import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_tVMC(g, *, s=1/2, total_sz=None):
    # a verifier hilbert for spin0_5_tVMC system: creates a spin-1/2 Hilbert space for a given graph
    # g: graph object representing the lattice structure (e.g., nk.graph.Square)
    # s: spin quantum number (default: 1/2)
    # total_sz: optional constraint on total magnetization (default: None)
    #
    # Creates a Spin Hilbert space object with spin quantum number s=1/2
    # and number of sites equal to the number of nodes in graph g
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi

# === statemodel_verifier_spin0_5_tVMC.py ===
import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_tVMC(*, kernel_init=None, seed=42):
    # a verifier statemodel for spin0_5_tVMC system: returns a Jastrow ansatz model for spin-1/2 systems
    # kernel_init: initializer for the Jastrow kernel matrix (default: normal distribution)
    # seed: random seed for parameter initialization (default: 42)
    #
    # Implements a short-range Jastrow ansatz with symmetric complex kernel matrix
    # as the variational model for tVMC calculations on spin-1/2 systems
    #
    if kernel_init is None:
        kernel_init = nn.initializers.normal()
    
    ma = nk.models.Jastrow(kernel_init=kernel_init)
    return ma

# === compset_verifier_spin0_5_tVMC.py ===
# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_tVMC(hi, ma, *, optimizer_name="Adam", n_samples=1000, seed=42):
    # a verifier compset for spin0_5_tVMC system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for spin-1/2 system
    # ma: variational model/ansatz (e.g., neural network model like RBM)
    # optimizer_name: name of optimizer to use (default "Adam")
    # n_samples: number of Monte Carlo samples to use (default 1000)
    
    # Create sampler using MetropolisLocal which flips spins one by one
    sa = nk.sampler.MetropolisLocal(hi)
    
    # Create variational quantum state using the sampler and model
    # Ensure n_samples is divisible by n_chains to avoid warnings
    n_samples = (n_samples // 16) * 16  # Make n_samples divisible by 16 (n_chains)
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    # Create optimizer based on specified name
    if optimizer_name == "Adam":
        op = nk.optimizer.Adam()
    elif optimizer_name == "Sgd":
        op = nk.optimizer.Sgd()
    elif optimizer_name == "Momentum":
        op = nk.optimizer.Momentum()
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    # Initialize the variational state parameters
    vs.init_parameters(seed=seed)
    
    return sa, op, vs

# === effector_verifier_spin0_5_tVMC.py ===
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
    # Create TDVP driver for time evolution using time-dependent variational principle
    # Create ODE solver for TDVP
    ode_solver = nk.experimental.dynamics.RK4  # Use RK4 solver from netket.experimental.dynamics
    
    driv = nk.experimental.TDVP(
        vs,           # Variational state
        op,           # Optimizer
        ha,           # Hamiltonian operator
        t0=t0,        # Initial time
        dt=dt,        # Time step
        ode_solver=ode_solver,  # ODE solver for integration
        propagation_type='real'  # Real-time dynamics
    )
    
    return driv

# === hamiltonian_verifier_spin0_5_tVMC.py ===
import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_tVMC(hi, g, *, J=1.0, h=1.0):
    # a verifier hamiltonian for spin0_5_tVMC system: transverse field Ising model with tunable parameters
    # hi: Hilbert space object for spin-1/2 system (nk.hilbert.Spin)
    # g: graph object defining lattice structure (nk.graph)
    # J: coupling strength for spin-spin interactions (default: 1.0)
    # h: strength of transverse magnetic field (default: 1.0)
    
    # Construct transverse field Ising model Hamiltonian
    # Using LocalOperator to build custom Hamiltonian
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h)
    
    return ha

# === observable_verifier_spin0_5_tVMC.py ===
import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_tVMC(hi, *, observable_type="magnetization", site_index=0):
    # a verifier observable for spin0_5_tVMC system: defines common observables for spin-1/2 tVMC calculations
    # hi: Hilbert space object defining the quantum system (e.g., nk.hilbert.Spin(0.5, N))
    # observable_type: type of observable to create (e.g., "magnetization", "energy", "custom")
    # site_index: index of the site for local observables (default: 0)
    #
    # Creates and returns an observable operator based on the specified type
    # For magnetization, returns sigma_x on specified site
    # For energy, returns the Hamiltonian operator
    # For custom, returns a zero operator as template
    #
    if observable_type == "magnetization":
        obs = nk.operator.spin.sigmax(hi, site_index)
    elif observable_type == "energy":
        # Energy observable would typically be the Hamiltonian
        # This would require additional parameters in practice
        obs = None  # Placeholder - Hamiltonian would need J, h parameters
    else:  # custom
        # Example of custom observable inheriting from AbstractObservable
        class ZeroOperator(nk.experimental.observable.AbstractObservable):
            @property
            def dtype(self):
                return float
            
            @property
            def hilbert(self):
                return hi
                
        obs = ZeroOperator()
    
    return obs


def main():
    # Define minimal parameters for quick verification
    L = 4  # Small lattice size (4 sites)
    g = nk.graph.Hypercube(length=L, n_dim=1)  # 1D chain with 4 sites
    hi = hilbert_verifier_spin0_5_tVMC(g)  # Create Hilbert space
    ma = statemodel_verifier_spin0_5_tVMC()  # Create variational model
    ma = statemodel_verifier_spin0_5_tVMC()  # Create variational model
    
    # Create compset components
    sa, op, vs = compset_verifier_spin0_5_tVMC(hi, ma, optimizer_name="Adam", n_samples=1000)
    
    # Create Hamiltonian
    ha = hamiltonian_verifier_spin0_5_tVMC(hi, g, J=1.0, h=1.0)
    
    # Create TDVP driver for time evolution
    driv = effector_verifier_spin0_5_tVMC(ha, vs, op, t0=0.0, dt=0.01)
    
    # Create an observable (magnetization on site 0)
    obs = observable_verifier_spin0_5_tVMC(hi, observable_type="magnetization", site_index=0)
    
    # Return all components to verify they were created
    return {
        'graph': g,
        'hilbert': hi,
        'model': ma,
        'sampler': sa,
        'optimizer': op,
        'variational_state': vs,
        'hamiltonian': ha,
        'driver': driv,
        'observable': obs
    }

if __name__ == "__main__":
    results = main()
    print("Verification completed successfully!")
    print(f"Created {len(results)} components for spin-1/2 tVMC system")

# === compset_verifier_spin0_5_tVMC.py ===
# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_tVMC(hi, ma, *, optimizer_name="Adam", n_samples=1000):
    # a verifier compset for spin0_5_tVMC system: creates sampler, optimizer, and variational quantum state
    # hi: Hilbert space object for spin-1/2 system
    # ma: variational model/ansatz (e.g., neural network model like RBM)
    # optimizer_name: name of optimizer to use (default "Adam")
    # n_samples: number of Monte Carlo samples to use (default 1000)
    
    # Create sampler using MetropolisLocal which flips spins one by one
    sa = nk.sampler.MetropolisLocal(hi)
    
    # Create variational quantum state using the sampler and model
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    # Create optimizer based on specified name
    if optimizer_name == "Adam":
        op = nk.optimizer.Adam()
    elif optimizer_name == "Sgd":
        op = nk.optimizer.Sgd()
    elif optimizer_name == "Momentum":
        op = nk.optimizer.Momentum()
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    return sa, op, vs

# === effector_verifier_spin0_5_tVMC.py ===
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
    # Create ODE solver for TDVP
    ode_solver = nk.dynamics.RK4()  # Use RK4 solver for time evolution
    
    driv = nk.experimental.TDVP(
        vs,           # Variational state
        op,           # Optimizer
        ha,           # Hamiltonian operator
        t0=t0,        # Initial time
        dt=dt,        # Time step
        ode_solver=ode_solver,  # ODE solver for integration
        propagation_type='real'  # Real-time dynamics
    )
    
    return driv

# === hamiltonian_verifier_spin0_5_tVMC.py ===
import netket as nk

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_tVMC(hi, g, *, J=1.0, h=1.0):
    # a verifier hamiltonian for spin0_5_tVMC system: transverse field Ising model with tunable parameters
    # hi: Hilbert space object for spin-1/2 system (nk.hilbert.Spin)
    # g: graph object defining lattice structure (nk.graph)
    # J: coupling strength for spin-spin interactions (default: 1.0)
    # h: strength of transverse magnetic field (default: 1.0)
    
    # Construct transverse field Ising model Hamiltonian
    # Using LocalOperator to build custom Hamiltonian
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=h)
    
    return ha

# === hilbert_verifier_spin0_5_tVMC.py ===
import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_tVMC(g, *, s=1/2, total_sz=None):
    # a verifier hilbert for spin0_5_tVMC system: creates a spin-1/2 Hilbert space for a given graph
    # g: graph object representing the lattice structure (e.g., nk.graph.Square)
    # s: spin quantum number (default: 1/2)
    # total_sz: optional constraint on total magnetization (default: None)
    #
    # Creates a Spin Hilbert space object with spin quantum number s=1/2
    # and number of sites equal to the number of nodes in graph g
    #
    hi = nk.hilbert.Spin(s=s, N=g.n_nodes, total_sz=total_sz)
    return hi

# === observable_verifier_spin0_5_tVMC.py ===
import netket as nk

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_tVMC(hi, *, observable_type="magnetization", site_index=0):
    # a verifier observable for spin0_5_tVMC system: defines common observables for spin-1/2 tVMC calculations
    # hi: Hilbert space object defining the quantum system (e.g., nk.hilbert.Spin(0.5, N))
    # observable_type: type of observable to create (e.g., "magnetization", "energy", "custom")
    # site_index: index of the site for local observables (default: 0)
    #
    # Creates and returns an observable operator based on the specified type
    # For magnetization, returns sigma_x on specified site
    # For energy, returns the Hamiltonian operator
    # For custom, returns a zero operator as template
    #
    if observable_type == "magnetization":
        obs = nk.operator.spin.sigmax(hi, site_index)
    elif observable_type == "energy":
        # Energy observable would typically be the Hamiltonian
        # This would require additional parameters in practice
        obs = None  # Placeholder - Hamiltonian would need J, h parameters
    else:  # custom
        # Example of custom observable inheriting from AbstractObservable
        class ZeroOperator(nk.experimental.observable.AbstractObservable):
            @property
            def dtype(self):
                return float
            
            @property
            def hilbert(self):
                return hi
                
        obs = ZeroOperator()
    
    return obs

# === statemodel_verifier_spin0_5_tVMC.py ===
import netket as nk

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_tVMC(*, kernel_init=None, seed=42):
    # a verifier statemodel for spin0_5_tVMC system: returns a Jastrow ansatz model for spin-1/2 systems
    # kernel_init: initializer for the Jastrow kernel matrix (default: normal distribution)
    # seed: random seed for parameter initialization (default: 42)
    #
    # Implements a short-range Jastrow ansatz with symmetric complex kernel matrix
    # as the variational model for tVMC calculations on spin-1/2 systems
    #
    if kernel_init is None:
        kernel_init = nk.nn.initializers.normal()
    
    ma = nk.models.Jastrow(kernel_init=kernel_init)
    return ma

