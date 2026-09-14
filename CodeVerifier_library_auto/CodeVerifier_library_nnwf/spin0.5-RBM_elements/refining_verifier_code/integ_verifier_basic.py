import netket as nk

# hilbert-----------------------------------------------------------------------------------------
def hilbert_verifier_spin0_5_RBM(g, *, s: float = 0.5, total_sz: float = None):
    # a verifier hilbert for spin0_5_RBM system: creates a spin-1/2 Hilbert space for RBM calculations
    # g: graph object defining the lattice structure and number of sites
    # s: spin value, set to 0.5 for spin-1/2 systems (default: 0.5)
    # total_sz: optional constraint for total magnetization (sum of spins), if None no constraint is applied
    #
    # Create spin-1/2 Hilbert space with N sites from graph
    if total_sz is not None:
        # Apply sum constraint for fixed total magnetization
        constraint = nk.constraints.SumConstraint(total_sz=total_sz)
        hi = nk.hilbert.Spin(s=s, N=g.n_nodes, constraint=constraint)
    else:
        # No constraint on total magnetization
        hi = nk.hilbert.Spin(s=s, N=g.n_nodes)
    #
    return hi

# statemodel-----------------------------------------------------------------------------------------
def statemodel_verifier_spin0_5_RBM(*, alpha: float = 1.0):
    # a verifier statemodel for spin0_5_RBM system: creates an RBM model for spin-1/2 systems
    # alpha: scaling factor for number of hidden units (hidden_units = alpha * visible_units), controls model capacity (default: 1.0)
    #
    # Creates a Restricted Boltzmann Machine (RBM) model with specified alpha parameter
    # The model is used with variational states like MCState for quantum simulations
    #
    ma = nk.models.RBM(alpha=alpha)
    return ma

# compset-----------------------------------------------------------------------------------------
def compset_verifier_spin0_5_RBM(hi, ma, *, n_samples: int = 1000, learning_rate: float = 0.01):
    # a verifier compset for spin0_5_RBM system: creates sampler, optimizer, and variational quantum state for spin0.5-RBM
    # hi: Hilbert space object for spin-1/2 system
    # ma: model/ansatz object (e.g., RBM)
    # n_samples: number of Monte Carlo samples for variational state (default: 1000)
    # learning_rate: learning rate for optimizer (default: 0.01)
    
    # Create sampler for spin0.5-RBM system
    sa = nk.sampler.MetropolisLocal(hi)
    
    # Construct variational quantum state with sampler and model
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    
    # Define optimizer for RBM wave function training
    op = nk.optimizer.Sgd(learning_rate=learning_rate)
    
    return sa, op, vs

# hamiltonian-----------------------------------------------------------------------------------------
def hamiltonian_verifier_spin0_5_RBM(hi, g, *, J=-1.0, h=1.0):
    # a verifier hamiltonian for spin0_5_RBM system: creates a Heisenberg or Ising Hamiltonian for spin-1/2 RBM models
    # hi: Hilbert space object for spin-1/2 system (nk.hilbert.Spin)
    # g: graph/lattice object defining the system geometry (nk.graph)
    # J: coupling parameter for spin interactions (default: -1.0)
    # h: transverse field parameter (default: 1.0)
    
    # Create Ising Hamiltonian with given parameters
    ha = nk.operator.Ising(hilbert=hi, graph=g, J=J, h=h)
    
    return ha

# effector-----------------------------------------------------------------------------------------
def effector_verifier_spin0_5_RBM(ha, op, vs, *, n_samples=1000, n_discard=None):
    # a verifier effector for spin0_5_RBM system: creates a VMC_SR driver for spin0.5-RBM calculations
    # ha: Hamiltonian operator (e.g., nk.operator.LocalOperator)
    # op: Optimizer (e.g., nk.optimizer.Adam)
    # vs: Variational state (e.g., nk.vqs.MCState)
    # n_samples: Number of samples for Monte Carlo estimation (default: 1000)
    # n_discard: Number of samples to discard for equilibration (default: n_samples/10)
    
    # Set default value for n_discard if not provided
    if n_discard is None:
        n_discard = n_samples // 10
    
    # Update the number of samples in the variational state
    vs.n_samples = n_samples
    
    # Create VMC driver with stochastic reconfiguration
    driv = nk.driver.VMC_SR(
        hamiltonian=ha,
        optimizer=op,
        variational_state=vs,
        diag_shift=0.01
    )
    
    return driv

# observable-----------------------------------------------------------------------------------------
def observable_verifier_spin0_5_RBM(hi, *, dtype=complex):
    # a verifier observable for spin0_5_RBM system: defines a custom observable for spin-1/2 RBM system
    # hi: Hilbert space object for spin-1/2 system (e.g., nk.hilbert.Spin)
    # dtype: data type for the observable (default: complex)
    #
    # Create a custom observable by subclassing AbstractObservable
    class CustomObservable(nk.experimental.observable.AbstractObservable):
        @property
        def dtype(self):
            return dtype
        
        @property
        def hilbert(self):
            return hi
            
    # Instantiate the observable with the provided Hilbert space
    obs = CustomObservable(hi)
    
    return obs

def main():
    # Define minimal parameters for quick execution
    N = 4  # Small system size (4 sites)
    alpha = 1.0  # Default RBM parameter
    n_samples = 100  # Reduced number of samples for faster execution
    learning_rate = 0.01  # Default learning rate
    J = -1.0  # Default coupling parameter
    h = 1.0  # Default transverse field
    
    # Create graph/lattice (1D chain with N sites)
    g = nk.graph.Hypercube(length=N, n_dim=1)
    
    # Create Hilbert space for spin-1/2 system
    hi = hilbert_verifier_spin0_5_RBM(g, s=0.5)
    
    # Create RBM model/ansatz
    ma = statemodel_verifier_spin0_5_RBM(alpha=alpha)
    
    # Create compset (sampler, optimizer, variational state)
    sa, op, vs = compset_verifier_spin0_5_RBM(hi, ma, n_samples=n_samples, learning_rate=learning_rate)
    
    # Create Hamiltonian
    ha = hamiltonian_verifier_spin0_5_RBM(hi, g, J=J, h=h)
    
    # Create effector (VMC driver)
    driv = effector_verifier_spin0_5_RBM(ha, op, vs, n_samples=n_samples)
    
    # Create observable
    obs = observable_verifier_spin0_5_RBM(hi)
    
    # Print summary of created objects
    print(f"Created {N}-site spin-1/2 system")
    print(f"Hilbert space: {hi}")
    print(f"RBM model with alpha={alpha}")
    print(f"Variational state with {n_samples} samples")
    print(f"Hamiltonian: Ising model with J={J}, h={h}")
    print(f"VMC driver created")
    print(f"Custom observable created")
    
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

# Run the main function if this script is executed directly
if __name__ == "__main__":
    results = main()
