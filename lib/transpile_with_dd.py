from qiskit import transpile
import qiskit.compiler.transpiler as transpiler
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager, level_1_pass_manager, level_2_pass_manager, level_3_pass_manager
from qiskit.transpiler.passes import DynamicalDecoupling
from qiskit.circuit.library import XGate

preset_pms = [level_0_pass_manager, level_1_pass_manager, level_2_pass_manager, level_3_pass_manager]

def _transpile_circuit(circuit_config_tuple):
    circuit, transpile_config = circuit_config_tuple

    pass_manager_config = transpile_config["pass_manager_config"]

    # we choose an appropriate one based on desired optimization level
    level = transpile_config["optimization_level"]

    pass_manager = preset_pms[level](pass_manager_config)
    
    default_passes = pass_manager.passes()
    scheduling = default_passes[-2]['passes']
    dd = DynamicalDecoupling(pass_manager_config.instruction_durations, dd_sequence=[XGate(), XGate()])
    scheduling.append(dd)
    pass_manager.replace(-2, scheduling)

    result = pass_manager.run(
        circuit, callback=transpile_config["callback"], output_name=transpile_config["output_name"]
    )
    
    return result

def transpile_with_dynamical_decoupling(circuits, backend=None, initial_layout=None, optimization_level=1):
    if not isinstance(circuits, list):
        return_singleton = True
        circuits = [circuits]
    else:
        return_singleton = False
    
    transpile_args = transpiler._parse_transpile_args(
        circuits, # circuits,
        backend, # backend,
        None, # basis_gates,
        None, # coupling_map,
        None, # backend_properties,
        initial_layout, # initial_layout,
        None, # layout_method,
        None, # routing_method,
        None, # translation_method,
        'alap', # scheduling_method,
        None, # instruction_durations,
        None, # dt,
        None, # approximation_degree,
        None, # seed_transpiler,
        optimization_level, # optimization_level,
        None, # callback,
        None, # output_name,
        None  # timing_constraints,
    )

    transpiler._check_circuits_coupling_map(circuits, transpile_args, backend)
    
    # Transpile circuits in parallel
    transpiled_circuits = transpiler.parallel_map(_transpile_circuit, list(zip(circuits, transpile_args)))
    
    if return_singleton:
        return transpiled_circuits[0]
    else:
        return transpiled_circuits
