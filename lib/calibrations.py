import re
import collections
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as scispec
import scipy.optimize as sciopt
from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit import Gate
from qiskit.result import Result
from qiskit.test.mock import FakeValencia
from qiskit.ignis.mitigation.measurement import complete_meas_cal, CompleteMeasFitter, MeasurementFilter
from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData, FitVal
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData
from qiskit_experiments.curve_analysis import plot_curve_fit, plot_errorbar, curve_fit
#from qiskit_experiments.curve_analysis.curve_fit import process_curve_data
from qiskit_experiments.curve_analysis.data_processing  import level2_probability, filter_data

from rttgen import find_native_cr_direction, LinearizedCR

class MeasurementErrorAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options()
    
    def _run_analysis(self, experiment_data, parameter_guess=None, plot=True, ax=None):
        state_labels = []
        for datum in experiment_data.data():
            state_label = datum['metadata']['state_label']
            if state_label in state_labels:
                break
            state_labels.append(state_label)

        meas_fitter = CompleteMeasFitter(None, state_labels, circlabel='mcal')
        
        nstates = len(state_labels)

        for job_id in experiment_data.job_ids:
            full_result = experiment_data.backend.retrieve_job(job_id).result()
            # full_result might contain repeated experiments
            for iset in range(len(full_result.results) // nstates):
                try:
                    date = full_result.date
                except:
                    date = None
                try:
                    status = full_result.status
                except:
                    status = None
                try:
                    header = full_result.header
                except:
                    header = None
                    
                result = Result(full_result.backend_name, full_result.backend_version, \
                                full_result.qobj_id, full_result.job_id, \
                                full_result.success, full_result.results[iset * nstates:(iset + 1) * nstates], \
                                date=date, status=status, header=header, **full_result._metadata)

                meas_fitter.add_data(result)
        
        results = [
            AnalysisResultData('error_matrix', meas_fitter.cal_matrix, extra=state_labels)
        ]
                
        plots = []
        if plot:
            figure, ax = plt.subplots(1, 1)
            meas_fitter.plot_calibration(ax=ax)
            plots.append(figure)
        
        return results, plots
    
class MeasurementErrorExperiment(BaseExperiment):
    __analysis_class__ = MeasurementErrorAnalysis
    
    def __init__(self, qubit_list, circuits_per_state=1):
        super().__init__(qubit_list)
        
        self.circuits_per_state = circuits_per_state

    def circuits(self, backend=None):
        if backend is None:
            backend = FakeValencia()
            print('Using FakeValencia for backend')
            
        qreg = QuantumRegister(len(self.physical_qubits))

        circuits, state_labels = complete_meas_cal(qubit_list=list(range(qreg.size)), qr=qreg, circlabel='mcal')
        for circuit, state_label in zip(circuits, state_labels):
            circuit.metadata = {
                'experiment_type': self._type,
                'physical_qubits': self.physical_qubits,
                'state_label': state_label
            }

        return circuits * self.circuits_per_state
    
class MeasurementErrorMitigation(object):
    def __init__(self, backend, qubits):
        self.backend = backend
        self.qubits = qubits
        self.filter = None
        
    def run_experiment(self, circuits_per_state=1):
        exp = MeasurementErrorExperiment(self.qubits, circuits_per_state=circuits_per_state)
        exp_data = exp.run(backend=self.backend, shots=self.backend.configuration().max_shots)
        print('Experiment ID:', exp_data.experiment_id)
        exp_data.block_for_results()
        exp_data.save()
        self._load_from_exp_data(exp_data)
        
    def load_matrix(self, experiment_id):
        exp_data = DbExperimentData.load(experiment_id, self.backend.provider().service("experiment"))
        self._load_from_exp_data(exp_data)
        
    def _load_from_exp_data(self, exp_data):
        analysis_result = exp_data.analysis_results()[0]
        self.filter = MeasurementFilter(analysis_result.value, analysis_result.extra)

    def apply(self, counts_list):
        corrected_counts = []
        for counts in counts_list:
            corrected_counts.append(self.filter.apply(counts))
        
        return corrected_counts


class LinearizedCRRabiAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options(
            parameter_guess={'alpha': np.pi * 0.5 / 500., 'phi0': 0., 'amp': 0.5, 'offset': 0.5}
        )
    
    def _run_analysis(self, experiment_data, parameter_guess=None, plot=True, ax=None):
        data = experiment_data.data()
        metadata = data[0]['metadata']
        
        counts = collections.defaultdict(int)
        totals = collections.defaultdict(int)
    
        for datum in data:
            width = datum['metadata']['xval']
            counts[width] += datum['counts'].get('00', 0)
            totals[width] += sum(datum['counts'].values())

        xdata = np.array(sorted(counts.keys()), dtype=float)
        ydata = np.array([counts[w] for w in sorted(counts.keys())], dtype=float)
        total = np.array([totals[w] for w in sorted(counts.keys())], dtype=float)
        ydata /= total
        ysigma = np.sqrt(ydata * (1. - ydata) / total)
        
        if parameter_guess is None:
            p0 = (np.pi * 0.5 / 500., 0., 0.5, 0.5)
        else:
            p0 = (parameter_guess['alpha'], parameter_guess['phi0'], parameter_guess['amp'], parameter_guess['offset'])
            
        def fun(x, alpha, phi0, amp, offset):
            return offset + amp * np.cos(alpha * x + phi0)
            
        fit_result = curve_fit(fun, xdata, ydata, p0, sigma=ysigma)
        
        if np.abs(fit_result.popt[1]) < np.pi / 4. and np.abs(fit_result.popt[2] - 0.5) < 0.1 and np.abs(fit_result.popt[3] - 0.5) < 0.1:
            quality = 'good'
        else:
            quality = 'bad'
            
        summary = {
            'shots_per_point': np.sum(total) / xdata.shape[0],
            'fit_result': fit_result
        }
        
        results = [
            AnalysisResultData('alpha', FitVal(fit_result.popt[0], fit_result.popt_err[0])),
            AnalysisResultData('phi0', FitVal(fit_result.popt[1], fit_result.popt_err[1])),
            AnalysisResultData('amp', FitVal(fit_result.popt[2], fit_result.popt_err[2])),
            AnalysisResultData('offset', FitVal(fit_result.popt[3], fit_result.popt_err[3])),
            AnalysisResultData('summary', summary, chisq=fit_result.reduced_chisq, quality=quality)
        ]

        plots = []
        if plot:
            ax = plot_curve_fit(fun, fit_result, ax=ax, fit_uncertainty=True)
            ax = plot_errorbar(xdata, ydata, ysigma, ax=ax)
            ax.tick_params(labelsize=14)
            ax.set_title('Rzx[{},{}]'.format(metadata['z_qubit'], metadata['x_qubit']))
            ax.set_xlabel('GaussianSquare width', fontsize=16)
            ax.set_ylabel('P(00)', fontsize=16)
            ax.grid(True)
            plots.append(ax.get_figure())
        
        return results, plots

class LinearizedCRRabiExperiment(BaseExperiment):
    __analysis_class__ = LinearizedCRRabiAnalysis
    
    def __init__(self, qubits, backend, max_width=1000, step_size=32, randomize_width=True, circuits_per_point=1):
        super().__init__(find_native_cr_direction(backend, qubits))
        
        self.width_values = np.arange(0, max_width + 1, step_size)
        if randomize_width:
            for i in range(self.width_values.shape[0]):
                self.width_values[i] += np.random.randint(-(step_size // 2), step_size // 2)
                
        self.circuits_per_point = circuits_per_point
                
    def _additional_metadata(self):
        return {'widths': self.width_values.tolist()}

    def circuits(self, backend=None):
        if backend is None:
            backend = FakeValencia()
            print('Using FakeValencia for backend')
            
        cr = LinearizedCR(backend, self.physical_qubits)
        
        circuits = []
        
        for width in self.width_values:
            circuit = cr.rzx_circuit(width)
            circuit.measure_all()
            circuit.metadata = {
                'experiment_type': self._type,
                'z_qubit': cr.z_qubit,
                'x_qubit': cr.x_qubit,
                'xval': width
            }            
            
            circuits.append(circuit)
        
        return circuits * self.circuits_per_point
