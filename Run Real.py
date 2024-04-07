import os

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np
from qiskit.circuit.library import IQP
from qiskit_ibm_runtime import Sampler, Options
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(channel="ibm_quantum",token="69f16f44de2970998b5b3a8dc5dc8622760f117523728ab8201ad4f07013b2ae8829944e812b7f453bc0ff943ae345b52ed356255a30c686ab3f52a8de2ca894")
backend = service.get_backend("ibm_kyoto")

program_files = sorted(os.listdir("programs"))
# qc = [QuantumCircuit.from_qasm_file(f"programs/{file}") for file in program_files]
# for c in qc:
#     c.remove_final_measurements()
#     c.measure_all()
#
# pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
# isa_circuit = pm.run(qc)
#
# options = Options()
# options.resilience_level = 0
# options.optimization_level = 0
# options.transpilation.skip_transpilation=True
#
# sampler = Sampler(backend=backend, options=options)
# job = sampler.run(isa_circuit,shots=4096)
# print(f">>> Job ID: {job.job_id()}")
# print(f">>> Job Status: {job.status()}")
job = service.job("cr1dxe58gdp0008g10eg")
result = job.result()
for program,dist in zip(program_files,result.quasi_dists):
    count = {}
    bit_dist = dist.binary_probabilities(2)
    for bit in bit_dist.keys():
        count[bit] = int(bit_dist[bit]*4096)

    print(program,count)