from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def make_Agate(theta, phi):
    circuit = QuantumCircuit(2, name='A')

    circuit.cx(1, 0)
    circuit.rz(-phi, 1)
    circuit.ry(-theta, 1)
    circuit.cx(0, 1)
    circuit.ry(theta, 1)
    circuit.rz(phi, 1)
    circuit.cx(1, 0)

    return circuit.to_instruction()

def make_pnp_ansatz(num_qubits, num_layers, initial_x_positions, structure=None, first_layer_structure=None, param_name=None, add_barriers=False):
    circuit  = QuantumCircuit(num_qubits)

    circuit.x(initial_x_positions)
    
    if add_barriers:
        circuit.barrier()
        
    if structure is None:
        structure = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
        structure += [(i, i + 1) for i in range(1, num_qubits - 1, 2)]
    
    if first_layer_structure is None:
        num_parameters = num_layers * len(structure) * 2
    else:
        num_parameters = (num_layers - 1) * len(structure) * 2 + len(first_layer_structure) * 2
        
    if param_name is None:
        param_name = '\N{greek small letter theta}'

    pv = ParameterVector(param_name, num_parameters)
    iparam = 0
        
    for ilayer in range(num_layers):
        if first_layer_structure is not None and ilayer == 0:
            layer_structure = first_layer_structure
        else:
            layer_structure = structure
            
        for qubits in layer_structure:
            Agate = make_Agate(*pv[iparam:iparam + 2])
            iparam += 2
            circuit.append(Agate, qubits)
                
        if add_barriers:
            circuit.barrier()

    return circuit
