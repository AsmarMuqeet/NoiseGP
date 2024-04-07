import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    phase_amplitude_damping_error
)
from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_distance

class Primitives:
    def __init__(self):
        self.noiseModel = NoiseModel()

    def dp_1q_ry(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = depolarizing_error(p,num_qubits=1)
        self.noiseModel.add_quantum_error(error,instructions=["ry"],qubits=[qubit],warnings=False)

    def dp_1q_rx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = depolarizing_error(p,num_qubits=1)
        self.noiseModel.add_quantum_error(error,instructions=["rx"],qubits=[qubit],warnings=False)

    def dp_1q_rz(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = depolarizing_error(p,num_qubits=1)
        self.noiseModel.add_quantum_error(error,instructions=["rz"],qubits=[qubit],warnings=False)

    def dp_1q_sx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = depolarizing_error(p,num_qubits=1)
        self.noiseModel.add_quantum_error(error,instructions=["sx"],qubits=[qubit],warnings=False)

    def dp_2q_cx(self):
        qubit1 = 0
        qubit2 = 1
        p = np.random.random()
        error = depolarizing_error(p,num_qubits=2)
        self.noiseModel.add_quantum_error(error,instructions=["cx"],qubits=[qubit1,qubit2],warnings=False)

    def ap_1q_ry(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = amplitude_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["ry"], qubits=[qubit],warnings=False)

    def ap_1q_rx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = amplitude_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["rx"], qubits=[qubit],warnings=False)

    def ap_1q_rz(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = amplitude_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["rz"], qubits=[qubit],warnings=False)

    def ap_1q_sx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = amplitude_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["sx"], qubits=[qubit],warnings=False)

    def ph_1q_ry(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["ry"], qubits=[qubit],warnings=False)

    def ph_1q_rx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["rx"], qubits=[qubit],warnings=False)

    def ph_1q_rz(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["rz"], qubits=[qubit],warnings=False)

    def ph_1q_sx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_damping_error(p)
        self.noiseModel.add_quantum_error(error, instructions=["sx"], qubits=[qubit],warnings=False)

    def pha_1q_ry(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_amplitude_damping_error(p,1-p)
        self.noiseModel.add_quantum_error(error, instructions=["ry"], qubits=[qubit],warnings=False)

    def pha_1q_rx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_amplitude_damping_error(p,1-p)
        self.noiseModel.add_quantum_error(error, instructions=["rx"], qubits=[qubit],warnings=False)

    def pha_1q_rz(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_amplitude_damping_error(p,1-p)
        self.noiseModel.add_quantum_error(error, instructions=["rz"], qubits=[qubit],warnings=False)

    def pha_1q_sx(self):
        qubit = np.random.randint(0, 2)
        p = np.random.random()
        error = phase_amplitude_damping_error(p,1-p)
        self.noiseModel.add_quantum_error(error, instructions=["sx"], qubits=[qubit],warnings=False)

    def __reset__(self):
        self.noiseModel = NoiseModel()


def get_random_model(depth=10):
    primitive = Primitives()
    method_list = [func for func in dir(primitive) if callable(getattr(primitive, func)) and not func.startswith("__")]
    random_funcs = np.random.choice(method_list, depth)
    primitive.__reset__()
    for func in random_funcs:
        primitive.__getattribute__(func)()

    return primitive.noiseModel

def get_random_individual():
    primitive = Primitives()
    method_list = [func for func in dir(primitive) if callable(getattr(primitive, func)) and not func.startswith("__")]
    random_funcs = np.random.choice(method_list, np.random.randint(1,5))
    primitive.__reset__()
    return [primitive.__getattribute__(func) for func in random_funcs]
def execute_circuit(circuit,noisemodel):
    sim = AerSimulator(noise_model=noisemodel)
    counts = sim.run(circuit,shots=4096).result().get_counts()
    return counts

def calculate_fitness(individual,benchmark):
    circuit = benchmark["circuit"]
    noise = benchmark["noise"]
    primitive = benchmark["primitive"]
    primitive.__reset__()
    for func in individual:
        func()
    new_model = primitive.noiseModel
    counts1 = execute_circuit(circuit,noise)
    counts2 = execute_circuit(circuit,new_model)
    if isinstance(counts1, list):
        avg_fitness = []
        for c1, c2 in zip(counts1, counts2):
            avg_fitness.append(hellinger_distance(c1, c2))

        fitness = np.mean(avg_fitness)
        return fitness
    else:
        fitness = hellinger_distance(counts1, counts2)
        return fitness

def calculate_fitness_real(individual,benchmark):
    circuit = benchmark["circuit"]
    primitive = benchmark["primitive"]
    primitive.__reset__()
    for func in individual:
        func()
    new_model = primitive.noiseModel
    counts1 = benchmark["real"]
    counts2 = execute_circuit(circuit,new_model)
    if isinstance(counts1, list):
        avg_fitness = []
        for c1, c2 in zip(counts1, counts2):
            avg_fitness.append(hellinger_distance(c1, c2))

        fitness = np.mean(avg_fitness)
        return fitness
    else:
        fitness = hellinger_distance(counts1, counts2)
        return fitness