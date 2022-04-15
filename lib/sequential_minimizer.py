import struct
import copy
import logging
import numpy as np
from qiskit import transpile
from qiskit.result.counts import Counts

logger = logging.getLogger('schwinger_rqd.sequential_minimizer')

def combine_counts(new, base):
    data = dict(base)
    for key, value in new.items():
        try:
            data[key] += value
        except KeyError:
            data[key] = value
                
    return Counts(data, time_taken=base.time_taken, creg_sizes=base.creg_sizes, memory_slots=base.memory_slots)

def array_to_hex(array):
    result = []
    if np.iscomplexobj(array):
        for x in array:
            result.append(struct.pack('!dd', x.real, x.imag).hex())
    else:
        for x in array:
            result.append(struct.pack('!d', x).hex())

    return tuple(result)

def hex_to_array(hexlist):
    if len(hexlist[0]) == 16:
        array = np.empty(len(hexlist), dtype=np.float64)
        for idx, h in enumerate(hexlist):
            x, = struct.unpack('!d', bytes.fromhex(h))
            array[idx] = x
    else:
        array = np.empty(len(hexlist), dtype=np.complex128)
        for idx, h in enumerate(hexlist):
            xr, xi = struct.unpack('!dd', bytes.fromhex(h))
            array[idx].real = xr
            array[idx].imag = xi
            
    return array

class SequentialVCMinimizer:
    def __init__(
        self,
        ansatz,
        cost_function,
        section_generators,
        backend,
        error_mitigation_filter=None,
        transpile_fn=transpile,
        transpile_options={'optimization_level': 1},
        shots_per_job=10000,
        num_jobs=20,
        run_options={},
        seed=12345
    ):
        self.cost_function = cost_function

        # Only the initial generators - can be changed from a callback
        self.cost_section_generators = list(section_generators)
        
        self._backend = backend
        self.shots_per_job = shots_per_job
        self.num_jobs = num_jobs
        self.run_options = dict(run_options)
        
        if self.shots_per_job > self._backend.configuration().max_shots:
            raise RuntimeError(f'shots_per_job must be smaller than {self._backend.configuration().max_shots}')
        
        self.error_mitigation_filter = error_mitigation_filter
        
        self.random_gen = np.random.default_rng(seed)

        self.callbacks_run = []
        self.callbacks_sweep = []
        
        self._ansatz = ansatz.copy()
        
        ansatz = ansatz.remove_final_measurements(inplace=False)
        if self._backend.name() != 'statevector_simulator':
            ansatz.measure_all(inplace=True)

        self._backend_ansatz = transpile_fn(ansatz, backend=self._backend, **transpile_options)

        self.state = None
        
    @property
    def ansatz(self):
        return self._ansatz
    
    @property
    def backend(self):
        return self._backend
    
    def switch_strategy(self, strategy, minimizer_params=None):
        self.state['strategy'] = strategy
        self._minimizer_state = None
        self._minimizer_params = copy.deepcopy(minimizer_params)

    def minimize(self,
        initial_param_val=None,
        strategy='sequential',
        minimizer_params=dict(),
        state=None,
        minimizer_state=None,
        max_sweeps=40,
        convergence_distance=1.e-3,
        convergence_cost=1.e-4
    ):
        if initial_param_val is not None:
            assert len(initial_param_val) == len(self._ansatz.parameters)
            self.state = {'param_val': initial_param_val, 'sweep': 0, 'cost': 1., 'shots': 0, 'strategy': strategy, 'raw_data': dict()}
            self._minimizer_state = None
            logger.info(f'Starting minimize() with initial parameter values {initial_param_val}')
        else:
            assert state is not None
            self.state = copy.deepcopy(state)
            self._minimizer_state = copy.deepcopy(minimizer_state)
            logger.info(f'Resuming minimization from parameter values {self.state["param_val"]}')
 
        self._minimizer_params = copy.deepcopy(minimizer_params)

        start_sweep = self.state['sweep']

        for isweep in range(start_sweep, max_sweeps):
            strategy = self.state['strategy']
            logger.info(f'Sweep {isweep} with strategy {strategy} - current cost {self.state["cost"]} current shots {self.state["shots"]}')
            
            self.state['sweep'] = isweep
            # values to be updated during sweep
            if 'initial_param_val' not in self.state:
                self.state['initial_param_val'] = np.copy(self.state['param_val'])
                self.state['initial_cost'] = self.state['cost']
                
            if strategy == 'sequential':
                self._minimize_sequential()

            elif strategy == 'gradient-descent':
                self._minimize_gradient_descent()
                
            elif strategy == 'largest-drop':
                self._minimize_largest_drop()
                
            for callback in self.callbacks_sweep:
                callback(self)
                
            distance = np.max(np.abs(self.state['param_val'] - self.state['initial_param_val']))
            cost_update = self.state['cost'] - self.state['initial_cost']
            
            self.state.pop('initial_param_val')
            self.state.pop('initial_cost')
            
            if distance < convergence_distance:
                logger.info('Minimization converged by parameter distance')
                break
                
            if abs(cost_update) < convergence_cost:
                logger.info('Minimization converged by cost update')
                break
                
        return self.state['param_val']

    def _minimize_sequential(self):
        if self._minimizer_state is None:
            self._minimizer_state = {'iparam': 0}

        param_ids = list(range(self._minimizer_state['iparam'], len(self._ansatz.parameters)))
        
        for iparam in param_ids:
            self._minimizer_state['iparam'] = iparam
            
            logger.debug(f'sequential: Calculating cost section for parameter {iparam}')
            cost_section = self._calculate_cost_section(iparam)
            
            theta_opt = cost_section.minimum()

            self.state['param_val'][iparam] = theta_opt
            self.state['cost'] = cost_section.fun(theta_opt)
            self.state['raw_data'] = dict() # memory cleanup
            
        self._minimizer_state = None
    
    def _minimize_gradient_descent(self):
        num_params = len(self._ansatz.parameters)
        
        if self._minimizer_state is None:
            self._minimizer_state = {
                'iparam': 0,
                'gradient': np.zeros(num_params, dtype='f8'),
                'minima': np.empty(num_params, dtype='f8')
            }
        
        param_ids = list(range(self._minimizer_state['iparam'], len(self._ansatz.parameters)))
        
        logger.info('gradient descent: Calculating cost sections for all parameters')

        for iparam in param_ids:
            self._minimizer_state['iparam'] = iparam
            
            cost_section = self._calculate_cost_section(iparam)
            
            self._minimizer_state['gradient'][iparam] = cost_section.grad()
            self._minimizer_state['minima'][iparam] = cost_section.minimum()

        steepest_direction = np.argmax(np.abs(self._minimizer_state['gradient']))
        step_size_steepest = (self._minimizer_state['minima'][steepest_direction] - self.state['param_val'][steepest_direction]) * 1. / 3.
        learning_rate = step_size_steepest / (-self._minimizer_state['gradient'][steepest_direction])
        parameter_shift = -self._minimizer_state['gradient'] * learning_rate

        self.state['param_val'] += parameter_shift
        self.state['cost'] -= self._minimizer_state['gradient'] @ parameter_shift # linear approximation
        self.state['raw_data'] = dict() # memory cleanup
        
        self._minimizer_state = None
    
    def _minimize_largest_drop(self):
        num_params = len(self._ansatz.parameters)
        
        if self._minimizer_state is None:
            self._minimizer_state = {
                'no_scouting_direction': -1,
                'iparam': 0,
                'largest_drop': 0.,
                'step_direction': -1
            }
        
        ## Scouting run
        logger.info('largest drop: Calculating cost sections for all parameters')

        shots_original = self.shots_per_job * self.num_jobs
        scouting_shots = shots_original // self._minimizer_params.get('scouting_factor', 10)
        self.num_jobs = scouting_shots // self.shots_per_job
        if scouting_shots % self.shots_per_job == 0:
            self.num_jobs += 1
        
        param_ids = list(range(self._minimizer_state['iparam'], num_params))
        try:
            param_ids.remove(self._minimizer_state['no_scouting_direction'])
        except ValueError:
            pass
        
        for iparam in param_ids:
            self._minimizer_state['iparam'] = iparam
            
            cost_section = self._calculate_cost_section(iparam)
            
            drop = cost_section.fun() - cost_section.fun(cost_section.minimum())
            if drop > self._minimizer_state['largest_drop']:
                self._minimizer_state['step_direction'] = iparam
                self._minimizer_state['largest_drop'] = drop
                
        self._minimizer_state['iparam'] = num_params

        self.num_jobs = shots_original // self.shots_per_job
        
        step_direction = self._minimizer_state['step_direction']

        if step_direction == -1:
            self._minimizer_state = None
            self.state['raw_data'] = dict()
            return
        
        logger.info(f'largest drop: Stepping in direction {step_direction}')
        
        cost_section = self._calculate_cost_section(step_direction, reuse=False)

        theta_opt = cost_section.minimum()
        
        self.state['param_val'][step_direction] = theta_opt
        self.state['cost'] = cost_section.fun(theta_opt)
        self.state['raw_data'] = dict()

        self._minimizer_state = {
            'no_scouting_direction': step_direction,
            'iparam': 0,
            'largest_drop': 0.,
            'step_direction': -1
        }
    
    def _make_circuit(self, iparam, test_value):
        param_val = np.copy(self.state['param_val'])
        param_val[iparam] = test_value
            
        param_dict = dict(zip(self._backend_ansatz.parameters, param_val))

        circuit = self._backend_ansatz.bind_parameters(param_dict)
            
        return circuit
    
    def _run_circuits(self, circuits, keys, reuse):
        """Run the circuits
        Args:
            circuits (list): List of circuits.
            shots (int): Total number of shots.
            header (dict): Header passed to callbacks.
        """

        if self._backend.name() == 'statevector_simulator':
            run_result = self._backend.run(circuits, **self.run_options).result()
            self.state['shots'] += self.shots_per_job * self.num_jobs * len(circuits)
            
            results = [res.data.statevector for res in run_result.results]
            
        else:
            results = list(Counts(dict()) for _ in range(len(circuits)))
            
            max_shots = 0
            circuits_to_run = list()
            circuit_indices = list()
            
            for ic, (circuit, key) in enumerate(zip(circuits, keys)):
                circuit_shots = self.shots_per_job * self.num_jobs

                try:
                    existing = self.state['raw_data'][key]
                except KeyError:
                    self.state['raw_data'][key] = Counts(dict())
                else:
                    results[ic] = combine_counts(existing, results[ic])
                    circuit_shots -= sum(existing.values())
                    if reuse or circuit_shots <= 0:
                        continue

                circuits_to_run.append(circuit)
                circuit_indices.append(ic)
                if circuit_shots > max_shots:
                    max_shots = circuit_shots
                    
            if circuits_to_run:
                run_options = dict(self.run_options)
                run_options['shots'] = self.shots_per_job

                num_jobs = max_shots // self.shots_per_job
                if max_shots % self.shots_per_job == 0:
                    num_jobs += 1

                for ijob in range(num_jobs):
                    run_result = self._backend.run(circuits_to_run, **run_options).result()
                    self.state['shots'] += self.shots_per_job * len(circuits_to_run)

                    for ic, counts in zip(circuit_indices, run_result.get_counts()):
                        results[ic] = combine_counts(counts, results[ic])
                        self.state['raw_data'][keys[ic]] = results[ic]

                    for callback in self.callbacks_run:
                        callback(self)
                    
        return results
    
    def _compute_probabilities(self, results):
        prob_dists = []
        
        if self._backend.name() == 'statevector_simulator':
            shots = self.shots_per_job * self.num_jobs
            for statevector in results:
                exact = np.square(np.abs(statevector))
                if shots <= 0:
                    prob_dists.append(exact)
                else:
                    prob_dist = self.random_gen.multinomial(shots, exact) / shots
                    prob_dists.append(prob_dist)
                
        else:
            if self.error_mitigation_filter is not None:
                corrected_counts = []
                for counts in results:
                    counts_dict = self.error_mitigation_filter.apply(counts)
                    corrected_counts.append(Counts(counts_dict, time_taken=counts.time_taken, creg_sizes=counts.creg_sizes, memory_slots=counts.memory_slots))
            else:
                corrected_counts = results

            for counts in corrected_counts:
                total = sum(counts.values())
                prob_dist = np.zeros(2 ** self._ansatz.num_qubits, dtype='f8')
                for idx, value in counts.int_outcomes().items():
                    prob_dist[idx] = value / total

                prob_dists.append(prob_dist)
                
        return prob_dists
    
    def _compute_costs(self, prob_dists, num_toys=0):
        costs = np.empty(len(prob_dists), dtype='f8')
        for iprob, prob_dist in enumerate(prob_dists):
            costs[iprob] = self.cost_function(prob_dist)
            
        if num_toys == 0:
            return costs
        
        sigmas = np.empty_like(costs)
        shots = self.shots_per_job * self.num_jobs
        
        for iprob, prob_dist in enumerate(prob_dists):
            sumw = 0.
            sumw2 = 0.
            for _ in range(num_toys):
                toy_prob_dist = self.random_gen.multinomial(shots, prob_dist) / shots
                toy_cost = self.cost_function(toy_prob_dist)
                sumw += toy_cost
                sumw2 += toy_cost * toy_cost
            
            mean = sumw / num_toys
            sigmas[iprob] = np.sqrt(sumw2 / num_toys - mean * mean)
            
        return costs, sigmas
    
    def _calculate_cost_section(
        self,
        iparam,
        reuse=True
    ):
        """Instantiate a cost section object and calculate the 1D section function from cost measurements.

        Args:
            iparam (int): Index of the parameter to compute the section over.
            reuse (bool): If True, results found in self.state['raw_data'] will be used. If False, new measurements
                are performed, and raw_data is updated.
        """
        logger.info(f'calculate cost section over parameter {iparam}')
        
        cost_section = self.cost_section_generators[iparam]()
        cost_section.set_thetas(self.state['param_val'][iparam])
        
        circuits = []
        keys = []
        for itheta, theta in enumerate(cost_section.thetas):
            circuits.append(self._make_circuit(iparam, theta))
            key = ''.join(array_to_hex(np.concatenate((self.state['param_val'][:iparam], [theta], self.state['param_val'][iparam + 1:]))))
            keys.append(key)
                
        results = self._run_circuits(circuits, keys, reuse)
        
        prob_dists = self._compute_probabilities(results)

        costs = self._compute_costs(prob_dists)

        cost_section.set_coeffs(costs)
            
        return cost_section
    

class SectionGenSwitcher:
    def __init__(self, generator, param_ids, threshold):
        self.generator = generator
        self.param_ids = param_ids
        self.threshold = threshold
        self.switched = False
        
    def callback_sweep(self, minimizer):
        if self.switched or minimizer.state['cost'] > self.threshold:
            return
        
        for iparam in self.param_ids:
            minimizer.cost_section_generators[iparam] = self.generator

        self.switched = True

from qiskit import Aer

class IdealCost:
    def __init__(self, ansatz):
        self.costs = []
        self.shots = []
        
        self._ansatz = transpile(ansatz, backend=Aer.get_backend('statevector_simulator'))
        
    def callback_sweep(self, minimizer):
        simulator = Aer.get_backend('statevector_simulator')
        
        param_dict = dict(zip(self._ansatz.parameters, minimizer.state['param_val']))
        circuit = self._ansatz.bind_parameters(param_dict)

        prob_dist = np.square(np.abs(simulator.run(circuit).result().data()['statevector']))
        cost = minimizer.cost_function(prob_dist)
        
        self.costs.append(cost)
        self.shots.append(minimizer.state['shots'])
