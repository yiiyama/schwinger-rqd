import re
import numpy as np

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate

def resolve_cx(backend, control_qubit, target_qubit):
    schedule = backend.defaults().instruction_schedule_map.get('cx', (control_qubit, target_qubit))
    
    cr_start_time = next(t for t, inst in schedule.instructions if type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel)
    last_cr_start, last_cr = next((t, inst) for t, inst in reversed(schedule.instructions) if type(inst) is pulse.Play and type(inst.channel) is pulse.ControlChannel)
    cr_end_time = last_cr_start + last_cr.pulse.duration

    pre_sched = schedule.filter(time_ranges=[(0, cr_start_time)])
    core_sched = pulse.Schedule(schedule.filter(instruction_types=[pulse.Play], time_ranges=[(cr_start_time, cr_end_time)]).shift(-cr_start_time), name='cx_core_{}_{}'.format(control_qubit, target_qubit))
    post_sched = schedule.filter(time_ranges=[(cr_end_time, schedule.duration)])
    
    channel_logic_map = {}
    for ch_name, ch_config in backend.configuration().channels.items():
        if ch_config['operates']['qubits'] == [control_qubit]:
            channel_logic_map[ch_name] = 0
        elif ch_config['operates']['qubits'] == [target_qubit]:
            channel_logic_map[ch_name] = 1
            
    def schedule_to_circuit(sched):
        circuit = QuantumCircuit(2)

        for _, inst in sched.instructions:
            try:
                qubit = channel_logic_map[inst.channel.name]
            except KeyError:
                continue
            
            if type(inst) is pulse.ShiftPhase and type(inst.channel) is pulse.DriveChannel:
                circuit.rz(-inst.phase, qubit)
            elif type(inst) is pulse.Play:
                matches = re.match('(X|Y)(|90)(p|m)_', inst.name)
                if not matches:
                    continue
                if matches.group(3) == 'm':
                    circuit.rz(np.pi, qubit)
                if matches.group(1) == 'Y':
                    circuit.rz(-np.pi / 2., qubit)
                if matches.group(2) == '':
                    circuit.x(qubit)
                else:
                    circuit.sx(qubit)
                if matches.group(1) == 'Y':
                    circuit.rz(np.pi / 2., qubit)
                if matches.group(3) == 'm':
                    circuit.rz(-np.pi, qubit)
                    
        return circuit

    pre_circ = schedule_to_circuit(pre_sched)
    post_circ = schedule_to_circuit(post_sched)
    return pre_circ, core_sched, post_circ

def cx_circuit(backend, control_qubit, target_qubit, use_rzx=False):
    circuit = QuantumCircuit(2)
    
    if backend.configuration().simulator:
        circuit.rz(np.pi / 2., 0)
        circuit.x(0)
        circuit.sx(1)
        circuit.rzx(np.pi / 4., 0, 1)
        circuit.x(0)
        circuit.rzx(-np.pi / 4., 0, 1)

        return circuit
        
    pre_circ, core_sched, post_circ = resolve_cx(backend, control_qubit, target_qubit)

    circuit.compose(pre_circ, inplace=True)
    
    if use_rzx:
        x_channel = next(inst for _, inst in core_sched.instructions if type(inst) is pulse.Play and type(inst.pulse) is pulse.GaussianSquare and type(inst.channel) is pulse.DriveChannel).channel
        x_qubit = None
        for ch_name, ch_config in backend.configuration().channels.items():
            if ch_name == x_channel.name:
                x_qubit = ch_config['operates']['qubits'][0]
                
        if x_qubit != control_qubit and x_qubit != target_qubit:
            raise RuntimeError('Something is wrong - x_qubit not in control or target')

        x_logical = 0 if x_qubit == control_qubit else 1
        z_logical = 1 if x_qubit == control_qubit else 0
        circuit.x(z_logical)
        circuit.rzx(np.pi / 2., z_logical, x_logical)
        
    else:
        core_gate = Gate(core_sched.name, 2, [])
        circuit.append(core_gate, (0, 1))
        circuit.add_calibration(core_gate.name, (control_qubit, target_qubit), core_sched)

    circuit.compose(post_circ, inplace=True)

    return circuit