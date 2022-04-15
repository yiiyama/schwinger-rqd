import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from hamiltonian import diagonalized_evolution, schwinger_model

def combined_counts(job, exp_block_size=1, mem_filter=None):
    raw_counts = job.result().get_counts()
    counts = raw_counts[:exp_block_size]
    for iexp, cdict in enumerate(raw_counts[exp_block_size:]):
        ic = iexp % exp_block_size
        for key, value in cdict.items():
            try:
                counts[ic][key] += value
            except KeyError:
                counts[ic][key] = value
                
    if mem_filter is not None:
        counts = mem_filter.apply(counts)

    return counts

def counts_to_probs(counts_list):
    """
    Args:
        counts_list (List[Dict[str, int]]): List of quantum experiment results, as given by job.result().get_counts()

    Returns:
        List[Tuple(ndarray, int)]: List of (probabilities, total_counts)
    """

    probs_list = []
    
    for counts in counts_list:
        nq = len(next(bitstring for bitstring in counts.keys()))
        probs = np.zeros((2 ** nq,), dtype='f8')
        
        for bitstring, count in counts.items():
            probs[int(bitstring, 2)] = count

        total = np.sum(probs)
        probs /= total
        
        probs_list.append((probs, total))

    return probs_list

def number_density(probs_list, num_toys=100):
    """
    Args: probs_list (List[Tuple(ndarray, int)]): List of (probabilities, total_counts)
    
    Returns:
        ndarray(T): Number density as a function of time
        ndarray(2, T): Number density lower/upper 1 sigma uncertainty as a function of time
    """

    nstates = probs_list[0][0].shape[0]
    nq = np.log2(nstates).astype(int)
    
    # even-index qubits: '0'='spin up'='occupied'='positron present'
    # odd-index qubits: '0'='spin up'='occupied'='electron absent'
    state_densities = np.zeros((nstates,), dtype='f8')
    for iq in range(nq):
        bit_up = (np.arange(nstates) >> iq) & 1
        if iq % 2 == 0:
            state_densities += (1 - bit_up) # bit up -> positron absent
        else:
            state_densities += bit_up # bit up -> electron present
        
    state_densities /= float(nq)
        
    densities = np.zeros((len(probs_list),), dtype='f8')
    uncertainties = np.zeros((2, len(probs_list)), dtype='f8')
    
    for itime, (probs, total) in enumerate(probs_list):
        densities[itime] = (state_densities @ probs)

        if total == 0:
            continue
            
        if num_toys > 0:
            # Run toy experiments
            toy_densities = np.empty((num_toys,), dtype='f8')
            for itoy in range(num_toys):
                toy_results = np.random.multinomial(total, probs) / total
                toy_densities[itoy] = state_densities @ toy_results

            toy_densities = np.sort(toy_densities)
            central_index = np.searchsorted(toy_densities, densities[itime])

            uncertainties[0, itime] = densities[itime] - toy_densities[int(central_index * 0.32)]
            uncertainties[1, itime] = toy_densities[central_index + int((num_toys - central_index) * 0.68)] - densities[itime]
        
    return densities, uncertainties
    
def insert_initial_counts(counts_list, initial_state):
    """Prepend a virtual 'counts' dictionary computed from the initial statevector to the counts list.
    
    Args:
        counts_list (List(Dict)): List of quantum experiment results, as given by Qiskit job.result().get_counts()
        initial_state (np.ndarray(shape=(2 ** num_spins), dtype=np.complex128)): Initial state vector.
    """
    
    num_bits = np.round(np.log2(initial_state.shape[0])).astype(int)

    initial_probs = np.square(np.abs(initial_state))
    fmt = '{{:0{}b}}'.format(num_bits)
    initial_counts = dict((fmt.format(idx), prob) for idx, prob in enumerate(initial_probs) if prob != 0.)

    return [initial_counts] + counts_list

def plot_curve(num_bits, aJ, am, omegat, initial_state=None):
    # omegat = t/2a

    hamiltonian = schwinger_model(num_bits, aJ, am)

    if initial_state is None:
        # Initial state as a statevector
        initial_state = np.zeros(2 ** num_bits, dtype=np.complex128)
        vacuum_state_index = 0
        for j in range(0, num_bits, 2):
            vacuum_state_index += (1 << j)
        initial_state[vacuum_state_index] = 1.

    # Plot the exact solution
    time_points_exact, statevectors = diagonalized_evolution(hamiltonian, initial_state, omegat)
    
    probs_exact = np.square(np.abs(statevectors)) # shape (D, T)
    probs_list_exact = [(probs_exact[:, itime], 0) for itime in range(probs_exact.shape[1])]
    densities_exact, _ = number_density(probs_list_exact)

    plt.plot(time_points_exact, densities_exact)
    
    return probs_list_exact

def plot_counts_with_curve(counts_list, num_bits, aJ, am, omegadt, num_tsteps, initial_state=None, num_toys=100):
    probs_list_exact = plot_curve(num_bits, aJ, am, omegadt * num_tsteps, initial_state=initial_state)

    # Plot the simulation results
    time_points = np.linspace(0., omegadt * num_tsteps, num_tsteps + 1, endpoint=True)
    probs_list = counts_to_probs(counts_list)
    
    probs_list.insert(0, probs_list_exact[0])
    densities, uncertainties = number_density(probs_list, num_toys=num_toys)

    plt.scatter(time_points, densities)
    if num_toys > 0:
        plt.errorbar(time_points, densities, yerr=uncertainties, elinewidth=1, linewidth=0)
