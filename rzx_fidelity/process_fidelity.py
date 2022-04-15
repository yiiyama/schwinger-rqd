import numpy as np
import matplotlib.pyplot as plt

from qiskit import IBMQ, Aer, QuantumCircuit
from qiskit.quantum_info import Operator, process_fidelity
from qiskit.ignis.verification.tomography import process_tomography_circuits, ProcessTomographyFitter

from qiskit_experiments.framework import BaseExperiment, BaseAnalysis, Options, AnalysisResultData
from qiskit_experiments.database_service import DbExperimentDataV1 as DbExperimentData

from calibrations import MeasurementErrorMitigation

class RzzFidelityAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls):
        return Options()
    
    def _run_analysis(self, experiment_data, phi_values, mem_exp_id):
        simulator = Aer.get_backend('qasm_simulator')
        
        mem = None
        if mem_exp_id is not None:
            mem = MeasurementErrorMitigation(experiment_data.backend, experiment_data.metadata['physical_qubits'])
            mem.load_matrix(mem_exp_id)
            
        fidelities = np.empty_like(phi_values)
            
        for iphi, phi_value in enumerate(phi_values):
            # Target circuit (use as dummy)
            target_circuit = QuantumCircuit(2)
            target_circuit.rzz(phi_value, 0, 1)
        
            target = Operator(target_circuit)
        
            dummy_circuits = process_tomography_circuits(target_circuit, target_circuit.qregs[0])
            dummy_result = simulator.run(dummy_circuits).result()
        
            fitter = ProcessTomographyFitter(dummy_result, dummy_circuits)
        
            for datum in experiment_data.data(slice(iphi * len(dummy_circuits), (iphi + 1) * len(dummy_circuits))):
                assert datum['metadata']['phi_index'] == iphi

                if mem:
                    counts = mem.filter.apply(datum['counts'])
                else:
                    counts = datum['counts']
            
                fitter._data[eval(datum['metadata']['basis'])] = counts
                
            channel = fitter.fit()
            fidelity = process_fidelity(channel, target=target, require_tp=False)

            fidelities[iphi] = fidelity
            
        results = [
            AnalysisResultData('fidelity', fidelities)
        ]
                
        figure, ax = plt.subplots(1, 1)
        ax.scatter(phi_values, fidelities, label=experiment_data.metadata['rttgen'])
        plots = [figure]
        
        return results, plots


class RzzFidelityExperiment(BaseExperiment):
    __analysis_class__ = RzzFidelityAnalysis
    
    def __init__(self, qubit_list, rttgen, phi_values, error_mitigation_exp_id=None):
        super().__init__(qubit_list)
        
        self.rttgen = rttgen
        self.phi_values = phi_values
        
        self.set_analysis_options(phi_values=self.phi_values, mem_exp_id=error_mitigation_exp_id)
        
    def _additional_metadata(self):
        return {'rttgen': type(self.rttgen).__name__, 'phi_values': self.phi_values}
        
    def circuits(self, backend):
        circuits = []
        
        for iphi, phi_value in enumerate(self.phi_values):
            rzz = self.rttgen.rzz_circuit(phi_value)

            for circuit in process_tomography_circuits(rzz, rzz.qregs[0]):
                circuit.metadata = {'phi': phi_value, 'phi_index': iphi, 'basis': circuit.name}
                circuits.append(circuit)

        return circuits