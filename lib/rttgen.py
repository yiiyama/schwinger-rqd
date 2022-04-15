import re
import numpy as np
import scipy.special as scispec
import scipy.optimize as sciopt

from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit import Gate, Parameter
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData

def get_closest_multiple_of_16(num):
    return int(np.round(num / 16.)) * 16

def get_instruction_by_name(schedule, pattern):
    return next(inst for _, inst in schedule.instructions if inst.name is not None and re.match(pattern, inst.name))

def find_native_cr_direction(backend, qubits):
    cx_schedule = backend.defaults().instruction_schedule_map.get('cx', qubits)
    gs_on_drive_channel = get_instruction_by_name(cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$')
    drive_channel = re.match(r'CR90p_(d[0-9]+)_u[0-9]+$', gs_on_drive_channel.name).group(1)
    x_qubit = backend.configuration().channels[drive_channel]['operates']['qubits'][0]
    z_qubit = qubits[0] if qubits[1] == x_qubit else qubits[1]

    return (z_qubit, x_qubit)

class BaseRtt(object):
    def __init__(self, backend, qubits):
#         if backend.configuration().simulator:
#             native_qubits = qubits
#         else:
#             native_qubits = find_native_cr_direction(backend, qubits)
        
#         self.native_direction = (native_qubits[0] == qubits[0])

        # Runtime backends don't expose defaults (and therefore the cx schedule)
        # We have to pass the qubits in the native order in the constructor
        self.native_direction = True
        
    def ryy_circuit(self, phi_value):
        rxx_circuit = self.rxx_circuit(phi_value)
    
        circuit = QuantumCircuit(rxx_circuit.qregs[0])
        circuit.sdg(0)
        circuit.sdg(1)
        circuit.compose(rxx_circuit, inplace=True)
        circuit.s(1)
        circuit.s(0)
    
        return circuit
    
    def rxxryy_circuit(self, phi_value):
        circuit = self.rxx_circuit(phi_value)
        circuit.compose(self.ryy_circuit(phi_value), inplace=True)

        return circuit
    
class CNOTBasedRtt(BaseRtt):
    def __init__(self, backend, qubits):
        super().__init__(backend, qubits)
        
        if self.native_direction:
            self.control_qubit = 0
            self.target_qubit = 1
        else:
            self.control_qubit = 1
            self.target_qubit = 0
    
    def rzx_circuit(self, phi_value):
        rzz_circuit = self.rzz_circuit(phi_value)
        
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(self.target_qubit)
        circuit.compose(rzz_circuit, inplace=True)
        circuit.h(self.target_qubit)

        return circuit
    
    def rzz_circuit(self, phi_value):
        circuit = QuantumCircuit(QuantumRegister(2))

        if phi_value == 0.:
            return circuit
        
        circuit.cx(self.control_qubit, self.target_qubit)
        circuit.rz(phi_value, self.target_qubit)
        circuit.cx(self.control_qubit, self.target_qubit)
    
        return circuit
    
    def rxx_circuit(self, phi_value):
        rzx_circuit = self.rzx_circuit(phi_value)
    
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(self.control_qubit)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.h(self.control_qubit)
    
        return circuit
    
    def rxxryy_circuit(self, phi_value):
        circuit = QuantumCircuit(QuantumRegister(2))
        
        if phi_value == 0.:
            return circuit

        qubits = (self.control_qubit, self.target_qubit)
        circuit.h(qubits)
        circuit.s(qubits)
        circuit.cx(*qubits)
        circuit.rx(phi_value, self.control_qubit)
        circuit.rz(phi_value, self.target_qubit)
        circuit.cx(*qubits)
        circuit.sdg(qubits)
        circuit.h(qubits)
        
        return circuit
    
class CRBasedRtt(BaseRtt):
    def __init__(self, backend, qubits):
        super().__init__(backend, qubits)
        
        if not backend.configuration().simulator:
            self.cx_schedule = backend.defaults().instruction_schedule_map.get('cx', qubits)
            
        self.backend = backend

        if self.native_direction:
            self.z_qubit = qubits[0]
            self.x_qubit = qubits[1]
        else:
            self.z_qubit = qubits[1]
            self.x_qubit = qubits[0]
    
    def get_xi_rzx(self, phi_value, circuit):
        xi_rzx = Gate('rzx_core_{}_{}({:.3f})'.format(self.z_qubit, self.x_qubit, phi_value), 2, [])
        
        cr_core_sched = self.get_cr_core_schedule(phi_value)
        circuit.add_calibration(xi_rzx.name, (self.z_qubit, self.x_qubit), cr_core_sched)
        
        return xi_rzx
    
    def rzx_circuit(self, phi_value):
        circuit = QuantumCircuit(QuantumRegister(2))

        if phi_value == 0.:
            return circuit
        
        xi_rzx = self.get_xi_rzx(phi_value, circuit)

        if self.native_direction:
            circuit.x(0)
            circuit.append(xi_rzx, (0, 1))
        else:
            circuit.h(0)
            circuit.h(1)
            circuit.x(1)
            circuit.append(xi_rzx, (1, 0))
            circuit.h(0)
            circuit.h(1)

        return circuit
    
    def rzz_circuit(self, phi_value):
        rzx_circuit = self.rzx_circuit(phi_value)
    
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(1)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.h(1)
    
        return circuit
    
    def rxx_circuit(self, phi_value):
        rzx_circuit = self.rzx_circuit(phi_value)
    
        circuit = QuantumCircuit(rzx_circuit.qregs[0])
        circuit.h(0)
        circuit.compose(rzx_circuit, inplace=True)
        circuit.h(0)
    
        return circuit
    

class LinearizedCR(CRBasedRtt):
    def __init__(self, backend, qubits):
        super().__init__(backend, qubits)
        
        self.alpha = 1.
        self.phi0 = 0.
        
    def load_calibration(self, experiment_id):
        load_exp = DbExperimentData.load(experiment_id, self.backend.provider().service("experiment"))
        self.alpha = next(res for res in load_exp.analysis_results() if res.name == 'alpha').value.value
        self.phi0 = next(res for res in load_exp.analysis_results() if res.name == 'phi0').value.value
    
    def get_cr_core_schedule(self, phi_value):
        width = (phi_value - self.phi0) / self.alpha
        
        with pulse.build(backend=self.backend):
            z_qubit_drive = pulse.drive_channel(self.z_qubit)
            x_qubit_drive = pulse.drive_channel(self.x_qubit)
            control_drive = pulse.control_channels(self.z_qubit, self.x_qubit)[0]

        x_pulse = get_instruction_by_name(self.cx_schedule, r'Xp_d[0-9]+$').pulse
        cx_cr_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_u[0-9]+$').pulse
        cx_rotary_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse

        if width == 0.:
            with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_core_{}_{}'.format(self.z_qubit, self.x_qubit)) as cr_core_sched:
                pulse.play(x_pulse, z_qubit_drive)

            return cr_core_sched

        cr_amp = cx_cr_pulse.amp
        crr_amp = cx_rotary_pulse.amp
        cr_sigma = cx_cr_pulse.sigma
        cr_flank_width = (cx_cr_pulse.duration - cx_cr_pulse.width) // 2

        # smallest multiple of 16 greater than |width| + 2 * (original flank width)
        #gs_duration = int(np.ceil((np.abs(width) + 2. * cr_flank_width) / 16.) * 16)
        gs_duration = get_closest_multiple_of_16(np.abs(width) + 2 * cr_flank_width)
        target_flank_width = (gs_duration - np.abs(width)) / 2.

        def area(sigma, flank_width):
            n_over_s2 = flank_width / sigma / np.sqrt(2.)
            gaus_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(n_over_s2)
            pedestal = np.exp(-n_over_s2 * n_over_s2)
            pedestal_integral = pedestal * flank_width
            return (gaus_integral - pedestal_integral) / (1. - pedestal)

        def diff_area(sigma, flank_width):
            n = flank_width / sigma
            n_over_s2 = n / np.sqrt(2.)
            gaus_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(n_over_s2)
            pedestal = np.exp(-n_over_s2 * n_over_s2)
            pedestal_integral = pedestal * flank_width
            diff_gaus_integral = gaus_integral / sigma - n * pedestal
            diff_pedestal_integral = n * n * n * pedestal
            return (diff_gaus_integral - diff_pedestal_integral - (diff_gaus_integral * pedestal_integral - gaus_integral * diff_pedestal_integral) / flank_width) / (1. - pedestal) / (1. - pedestal)

        cr_flank_area = area(cr_sigma, cr_flank_width)

        def func(sigma):
            return area(sigma, target_flank_width) - cr_flank_area

        def fprime(sigma):
            return diff_area(sigma, target_flank_width)

        gs_sigma = sciopt.newton(func, cr_sigma, fprime)
        
        phi_label = int(np.round(phi_value / np.pi * 180.))

        def gaus_sq(amp, rotary):
            if rotary:
                name = 'CRGS{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index)
            else:
                name = 'CRGS{}p_u{}'.format(phi_label, control_drive.index)

            return pulse.GaussianSquare(duration=gs_duration, amp=amp, sigma=gs_sigma, width=np.abs(width), name=name)

        def gaus(amp, rotary):
            if rotary:
                name = 'CRG{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index)
            else:
                name = 'CRG{}p_u{}'.format(phi_label, control_drive.index)

            return pulse.Gaussian(duration=(2 * cr_flank_width), amp=amp, sigma=cr_sigma, name=name)

        if width > 0.:
            forward = gaus_sq
            cancel = gaus
        else:
            forward = gaus
            cancel = gaus_sq

        cr_pulse = forward(cr_amp, False)
        cr_rotary_pulse = forward(crr_amp, True)
        cr_echo = forward(-cr_amp, False)
        cr_rotary_echo = forward(-crr_amp, True)

        cancel_pulse = cancel(-cr_amp, False)
        cancel_rotary_pulse = cancel(-crr_amp, True)
        cancel_echo = cancel(cr_amp, False)
        cancel_rotary_echo = cancel(crr_amp, True)

        with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_core_{}_{}'.format(self.z_qubit, self.x_qubit)) as cr_core_sched:
            ## echo (without the first X on control)
            with pulse.align_left():
                pulse.play(cr_echo, control_drive)
                pulse.play(cancel_echo, control_drive)
                pulse.play(cr_rotary_echo, x_qubit_drive)
                pulse.play(cancel_rotary_echo, x_qubit_drive)

            pulse.play(x_pulse, z_qubit_drive)

            ## forward
            with pulse.align_left():
                pulse.play(cr_pulse, control_drive)
                pulse.play(cancel_pulse, control_drive)
                pulse.play(cr_rotary_pulse, x_qubit_drive)
                pulse.play(cancel_rotary_pulse, x_qubit_drive)

        return cr_core_sched
    
    
class PulseEfficientCR(CRBasedRtt):
    def get_cr_core_schedule(self, phi_value):
        with pulse.build(backend=self.backend):
            z_qubit_drive = pulse.drive_channel(self.z_qubit)
            x_qubit_drive = pulse.drive_channel(self.x_qubit)
            control_drive = pulse.control_channels(self.z_qubit, self.x_qubit)[0]

        x_pulse = get_instruction_by_name(self.cx_schedule, r'Xp_d[0-9]+$').pulse
        cx_cr_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_u[0-9]+$').pulse
        cx_rotary_pulse = get_instruction_by_name(self.cx_schedule, r'CR90p_d[0-9]+_u[0-9]+$').pulse    

        if phi_value == 0.:
            with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
                pulse.play(x_pulse, z_qubit_drive)

            return cr_core_sched

        cr_amp = cx_cr_pulse.amp
        crr_amp = cx_rotary_pulse.amp
        sigma = cx_cr_pulse.sigma
        flank_width = (cx_cr_pulse.duration - cx_cr_pulse.width) // 2

        normal_flank_integral = np.sqrt(np.pi / 2.) * sigma * scispec.erf(flank_width / np.sqrt(2.) / sigma)
        pedestal = np.exp(-0.5 * np.square(flank_width / sigma))
        grounded_flank_integral = (normal_flank_integral - pedestal * flank_width) / (1. - pedestal)
        flank_area = np.abs(cr_amp) * grounded_flank_integral
        cr45_area_norm = np.abs(cr_amp) * cx_cr_pulse.width + 2. * flank_area
        minimum_phi = 2. * np.pi / 4. * (2. * flank_area) / cr45_area_norm

        phi_label = int(np.round(phi_value / np.pi * 180.))
        
        if phi_value <= minimum_phi:
            amp_ratio = phi_value / minimum_phi
            duration = 2 * flank_width
            cr_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * cr_amp), sigma=sigma, name='CR{}p_u{}'.format(phi_label, control_drive.index))
            cr_rotary_pulse = pulse.Gaussian(duration=duration, amp=(amp_ratio * crr_amp), sigma=sigma, name='CR{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
            cr_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * cr_amp), sigma=sigma, name='CR{}m_u{}'.format(phi_label, control_drive.index))
            cr_rotary_echo = pulse.Gaussian(duration=duration, amp=-(amp_ratio * crr_amp), sigma=sigma, name='CR{}m_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
        else:
            area = phi_value / 2. / (np.pi / 4.) * cr45_area_norm
            width = (area - 2. * flank_area) / np.abs(cr_amp)
            duration = get_closest_multiple_of_16(width + 2 * flank_width)
            cr_pulse = pulse.GaussianSquare(duration=duration, amp=cr_amp, sigma=sigma, width=width, name='CR{}p_u{}'.format(phi_label, control_drive.index))
            cr_rotary_pulse = pulse.GaussianSquare(duration=duration, amp=crr_amp, sigma=sigma, width=width, name='CR{}p_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))
            cr_echo = pulse.GaussianSquare(duration=duration, amp=-cr_amp, sigma=sigma, width=width, name='CR{}m_u{}'.format(phi_label, control_drive.index))
            cr_rotary_echo = pulse.GaussianSquare(duration=duration, amp=-crr_amp, sigma=sigma, width=width, name='CR{}m_d{}_u{}'.format(phi_label, x_qubit_drive.index, control_drive.index))

        with pulse.build(backend=self.backend, default_alignment='sequential', name='cr_gate_core') as cr_core_sched:
            ## echo (without the first X on control)
            with pulse.align_left():
                pulse.play(cr_echo, control_drive, name=cr_echo.name)
                pulse.play(cr_rotary_echo, x_qubit_drive, name=cr_rotary_echo.name)

            pulse.play(x_pulse, z_qubit_drive, name=x_pulse.name)

            ## forward
            with pulse.align_left():
                pulse.play(cr_pulse, control_drive, name=cr_pulse.name)
                pulse.play(cr_rotary_pulse, x_qubit_drive, name=cr_rotary_pulse.name)

        return cr_core_sched