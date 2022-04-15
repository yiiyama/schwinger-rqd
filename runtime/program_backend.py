import sys
import re
import json

from qiskit import IBMQ
from qiskit.providers.ibmq.runtime import UserMessenger, ProgramBackend

def get_instruction_by_name(schedule, pattern):
    return next(inst for _, inst in schedule.instructions if inst.name is not None and re.match(pattern, inst.name))

def find_native_cr_direction(backend, qubits):
    cx_schedule = backend.defaults().instruction_schedule_map.get('cx', qubits)
    gs_on_drive_channel = get_instruction_by_name(cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$')
    drive_channel = re.match(r'CR90p_(d[0-9]+)_u[0-9]+$', gs_on_drive_channel.name).group(1)
    x_qubit = backend.configuration().channels[drive_channel]['operates']['qubits'][0]
    z_qubit = qubits[0] if qubits[1] == x_qubit else qubits[1]

    return (z_qubit, x_qubit)

def main(backend, user_messenger, **kwargs):
    config = backend.configuration()
    if config is not None:
        config = config.to_dict()

    properties = backend.properties()
    if properties is not None:
        properties = properties.to_dict()
        
    status = backend.status()
    if status is not None:
        status = status.to_dict()
        
    try:
        account_provider = IBMQ.enable_account(kwargs['api_token'], hub=kwargs['hub'], group=kwargs['group'], project=kwargs['project'])
    except:
        account_provider = IBMQ.get_provider(hub=kwargs['hub'], group=kwargs['group'], project=kwargs['project'])
        
    ibmq_backend = account_provider.get_backend(backend.name())
    z_qubit, x_qubit = find_native_cr_direction(ibmq_backend, kwargs['qubits'])
    
    backend_dump = {
        'name': backend.name(),
        'configuration': config,
        'properties': properties,
        'provider class': type(backend.provider()).__name__,
        'status': status,
        'options': repr(backend.options),
        'native_direction': (z_qubit, x_qubit)
    }

    return backend_dump
