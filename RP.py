import pickle

import numpy as np
from qiskit import transpile, QuantumCircuit
from tqdm import tqdm

from primitives import *
from func_timeout import func_timeout, FunctionTimedOut
from qiskit_ibm_runtime import QiskitRuntimeService


def run_rp(circuit_file, noise_file,iterations=10):
    if isinstance(circuit_file,list):
        qc = [QuantumCircuit.from_qasm_file(f"programs/{cf}") for cf in circuit_file]
    else:
        qc = QuantumCircuit.from_qasm_file(f"programs/{circuit_file}")

    qc = transpile(qc, basis_gates=["sx", "id", "rx", "ry", "rz", "cx"])
    file = open(f"models/{noise_file}", "rb")
    params = pickle.load(file)
    file.close()
    noise = params["model"]
    run_results = []
    for run in tqdm(range(iterations)):
        primitive = Primitives()
        benchmark = {"circuit": qc, "primitive": primitive, "noise": noise}
        best = 1
        for i in range(300 * 20):
            individual = get_random_individual()
            try:
                fitness = func_timeout(timeout=5,func=calculate_fitness, args=(individual, benchmark))
            except FunctionTimedOut:
                fitness = 1
            best = min(best, fitness)
        run_results.append(best)

    if isinstance(circuit_file,list):
        file = open(f"results/{noise_file}_RP.pkl", "wb")
        pickle.dump(run_results, file)
        file.close()
    else:
        file = open(f"results/{circuit_file}_{noise_file}_RP.pkl", "wb")
        pickle.dump(run_results,file)
        file.close()

def run_rp_real(circuit_file, noise_file,iterations=10):

    service = QiskitRuntimeService(channel="ibm_quantum",
                                   token="69f16f44de2970998b5b3a8dc5dc8622760f117523728ab8201ad4f07013b2ae8829944e812b7f453bc0ff943ae345b52ed356255a30c686ab3f52a8de2ca894")

    job = service.job("cr1dxe58gdp0008g10eg")
    result = job.result()
    real = []
    for dist in result.quasi_dists:
        count = {}
        bit_dist = dist.binary_probabilities(2)
        for bit in bit_dist.keys():
            count[bit] = int(bit_dist[bit] * 4096)
        real.append(count)

    if isinstance(circuit_file,list):
        qc = [QuantumCircuit.from_qasm_file(f"programs/{cf}") for cf in circuit_file]
    else:
        qc = QuantumCircuit.from_qasm_file(f"programs/{circuit_file}")

    qc = transpile(qc, basis_gates=["sx", "id", "rx", "ry", "rz", "cx"])
    run_results = []
    for run in tqdm(range(iterations)):
        primitive = Primitives()
        benchmark = {"circuit": qc, "primitive": primitive, "real": real}
        best = 1
        for i in range(300 * 20):
            individual = get_random_individual()
            try:
                fitness = func_timeout(timeout=5,func=calculate_fitness_real, args=(individual, benchmark))
            except FunctionTimedOut:
                fitness = 1
            best = min(best, fitness)
        run_results.append(best)

    if isinstance(circuit_file,list):
        file = open(f"results/{noise_file}_RP.pkl", "wb")
        pickle.dump(run_results, file)
        file.close()
    else:
        file = open(f"results/{circuit_file}_{noise_file}_RP.pkl", "wb")
        pickle.dump(run_results,file)
        file.close()




if __name__ == '__main__':
    qc = QuantumCircuit.from_qasm_file("programs/ae_indep_qiskit_2.qasm")
    qc = transpile(qc,basis_gates=["sx","id","rx","ry","rz","cx"])
    file = open("models/result_3.pkl","rb")
    params = pickle.load(file)
    noise = params["model"]
    #noise = get_random_model()
    primitive = Primitives()
    benchmark = {"circuit":qc,"primitive":primitive,"noise":noise}
    best = 1
    for i in tqdm(range(300*40)):
        individual = get_random_individual()
        fitness = calculate_fitness(individual,benchmark)
        best = min(best,fitness)
    print(best)
