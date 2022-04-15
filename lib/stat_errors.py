import json
import multiprocessing
import numpy as np
import h5py

from qiskit import Aer, IBMQ, transpile
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
#from qiskit.providers.ibmq.runtime import UserMessenger

from main import make_step_circuits, run_forward_circuit, main
from observables import plot_counts_with_curve
from trotter import trotter_step_circuits
from rqd import make_ansatz, FISCPNP, FullOrderMatrix, QuadraticMatrix, LinearMatrix, QuadraticFit, LinearFit, FullOrderFit

max_sweeps = 10

def do_experiment(seed, queue, fit=False, shots=(8 * 8192)):
    rand = np.random.default_rng(seed)
    
    backend = Aer.get_backend('statevector_simulator')

    kwargs = {
        'num_site': 4,
        'aJ': 1.,
        'am': 0.5,
        'omegadt': 0.2,
        'num_tstep': 8,
        'tstep_per_rqd': 2,
        'error_matrix': np.eye(4, dtype='f8'),
        'maxfuncall': 100
    }

    num_site = kwargs['num_site']
    aJ = kwargs['aJ']
    am = kwargs['am']
    omegadt = kwargs['omegadt']
    num_tstep = kwargs['num_tstep']

    physical_qubits = kwargs.get('physical_qubits', None)

    single_step_circuit, two_step_circuit = make_step_circuits(num_site, aJ, am, omegadt, backend, physical_qubits)

    target_circuits = trotter_step_circuits(num_tstep, single_step_circuit, two_step_circuit, initial_state=None, measure=False)

    ansatz, params = make_ansatz(num_site // 2, num_site, num_site // 2, False)

    target_circuit = target_circuits[1]
    
    if fit:
        default_slice_gen = lambda: FullOrderFit(8)
        fiscpnp = FISCPNP(target_circuit, (ansatz, params), 1., backend, physical_qubits=physical_qubits, shots=8192, num_experiments=1, default_slice_gen=default_slice_gen, seed=seed)
        fiscpnp.cost_slice_generators[0] = QuadraticFit(8)
        fiscpnp.cost_slice_generators[1] = LinearFit(8)
        fiscpnp.cost_slice_generators[2] = QuadraticFit(8)
        fiscpnp.cost_slice_generators[3] = LinearFit(8)
        fiscpnp.cost_slice_generators[5] = LinearFit(8)
    else:
        fiscpnp = FISCPNP(target_circuit, (ansatz, params), 0., backend, physical_qubits=physical_qubits, shots=8*8192, num_experiments=1, default_slice_gen=FullOrderMatrix, seed=seed)
        fiscpnp.cost_slice_generators[0] = QuadraticMatrix()
        fiscpnp.cost_slice_generators[1] = LinearMatrix()
        fiscpnp.cost_slice_generators[2] = QuadraticMatrix()
        fiscpnp.cost_slice_generators[3] = LinearMatrix()
        fiscpnp.cost_slice_generators[5] = LinearMatrix()

    initial_params = rand.random(len(params)) * 2. * np.pi
    param_val, cost_values, ncall = fiscpnp.smo(initial_params, max_sweeps=max_sweeps, callbacks_step=[fiscpnp.ideal_cost_step])
    
    queue.put((param_val, cost_values, ncall, fiscpnp.ideal_costs_step))


if __name__ == '__main__':
    num_exp = 20
    num_params = 12
    
    seeds = np.random.randint(1024, 65536, size=num_exp)
    
    for exp_name, exp_conf in [('analytic', {'fit': False, 'shots': 8 * 8192}), ('fit_8_8k', {'fit': True, 'shots': 8192}), ('fit_4_16k', {'fit': True, 'shots': 2 * 8192}), ('fit_2_32k', {'fit': True, 'shots': 4 * 8192}), ('fit_8_64k', {'fit': True, 'shots': 8 * 8192})]:
        print('Starting experiment', exp_name)
        
        estimates = []
        cost_evolutions = []
        ideal_cost_evolutions = []
        funcalls = []

        procs = []
        queue = multiprocessing.Queue()
        for seed in seeds:
            proc = multiprocessing.Process(target=do_experiment, args=(seed, queue), kwargs=exp_conf)
            proc.start()
            procs.append(proc)

        for _ in range(num_exp):
            data = queue.get()
            estimates.append(data[0])
            cost_evolutions.append(data[1])
            funcalls.append(data[2])
            ideal_cost_evolutions.append(data[3])
            print('Received data from {} experiments'.format(len(estimates)))

        for proc in procs:
            proc.join()

        with h5py.File('stat_errors_{}.h5'.format(exp_name), 'w') as out:
            dataset = out.create_dataset('estimates', (num_exp, num_params), dtype='f8')
            for iest, est in enumerate(estimates):
                dataset[iest] = est

            dataset = out.create_dataset('costs', (num_exp, max_sweeps * num_params), dtype='f8')
            dataset[:] = 0.
            for iev, evol in enumerate(cost_evolutions):
                for istep, cost in enumerate(evol):
                    dataset[iev, istep] = cost
                    
            dataset = out.create_dataset('ideal_costs', (num_exp, max_sweeps * num_params), dtype='f8')
            dataset[:] = 0.
            for iev, evol in enumerate(ideal_cost_evolutions):
                for istep, cost in enumerate(evol):
                    dataset[iev, istep] = cost

            dataset = out.create_dataset('num_sweeps', (num_exp,), dtype='i')
            for iev, evol in enumerate(cost_evolutions):
                dataset[iev] = len(evol) // num_params

            dataset = out.create_dataset('num_calls', (num_exp,), dtype='i')
            for ic, ncalls in enumerate(funcalls):
                dataset[ic] = ncalls
