from qiskit import QuantumCircuit, transpile

def trotter_step_circuits(num_steps, circuit_elements, initial_state=None, measure=True):
    nsites = circuit_elements[0].num_qubits
    
    circuits = []
    for nrep in range(1, num_steps + 1):
        circuit = QuantumCircuit(nsites)
        if initial_state is None:
            circuit.x(range(0, nsites, 2))
        else:
            circuit.compose(initial_state, inplace=True)

        if len(circuit_elements) == 1:
            for _ in range(nrep):
                circuit.compose(circuit_elements[0], inplace=True)
        else:
            block_size = len(circuit_elements)
            while True:
                for _ in range(nrep // block_size):
                    circuit.compose(circuit_elements[block_size - 1], inplace=True)
                
                nrep = nrep % block_size
                if nrep == 0:
                    break

                block_size -= 1
                while circuit_elements[block_size - 1] is None:
                    block_size -= 1

        if measure:    
            circuit.measure_all(circuit.qregs[0])
            
        circuits.append(circuit)
    
    return circuits
