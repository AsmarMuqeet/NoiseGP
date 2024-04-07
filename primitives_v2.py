import random

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

class myNoiseModel(object):
    def __init__(self):
        self.noise = NoiseModel()

    def __reset__(self):
        self.noise = NoiseModel()

    def dp_1q_ry(self, arity, qubit, p):
        error = depolarizing_error(p, num_qubits=1)
        self.noise.add_quantum_error(error, instructions=["ry"], qubits=[qubit], warnings=False)
        return 0


    def dp_1q_rx(self, arity, qubit, p):
        error = depolarizing_error(p, num_qubits=1)
        self.noise.add_quantum_error(error, instructions=["rx"], qubits=[qubit], warnings=False)
        return 0


    def dp_1q_rz(self, arity, qubit, p):
        error = depolarizing_error(p, num_qubits=1)
        self.noise.add_quantum_error(error, instructions=["rz"], qubits=[qubit], warnings=False)
        return 0


    def dp_1q_sx(self, arity, qubit, p):
        error = depolarizing_error(p, num_qubits=1)
        self.noise.add_quantum_error(error, instructions=["sx"], qubits=[qubit], warnings=False)
        return 0


    def dp_2q_cx(self, arity, p):
        qubit1 = 0
        qubit2 = 1
        error = depolarizing_error(p, num_qubits=2)
        self.noise.add_quantum_error(error, instructions=["cx"], qubits=[qubit1, qubit2], warnings=False)
        return 0


    def ap_1q_ry(self, arity, qubit, p):
        error = amplitude_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["ry"], qubits=[qubit], warnings=False)
        return 0


    def ap_1q_rx(self, arity, qubit, p):
        error = amplitude_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["rx"], qubits=[qubit], warnings=False)
        return 0


    def ap_1q_rz(self, arity, qubit, p):
        error = amplitude_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["rz"], qubits=[qubit], warnings=False)
        return 0


    def ap_1q_sx(self, arity, qubit, p):
        error = amplitude_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["sx"], qubits=[qubit], warnings=False)
        return 0


    def ph_1q_ry(self, arity, qubit, p):
        error = phase_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["ry"], qubits=[qubit], warnings=False)
        return 0


    def ph_1q_rx(self, arity, qubit, p):
        error = phase_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["rx"], qubits=[qubit], warnings=False)
        return 0


    def ph_1q_rz(self, arity, qubit, p):
        error = phase_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["rz"], qubits=[qubit], warnings=False)
        return 0


    def ph_1q_sx(self, arity, qubit, p):
        error = phase_damping_error(p)
        self.noise.add_quantum_error(error, instructions=["sx"], qubits=[qubit], warnings=False)
        return 0


    def pha_1q_ry(self, arity, qubit, p):
        error = phase_amplitude_damping_error(p, 1 - p)
        self.noise.add_quantum_error(error, instructions=["ry"], qubits=[qubit], warnings=False)
        return 0


    def pha_1q_rx(self, arity, qubit, p):
        error = phase_amplitude_damping_error(p, 1 - p)
        self.noise.add_quantum_error(error, instructions=["rx"], qubits=[qubit], warnings=False)
        return 0


    def pha_1q_rz(self, arity, qubit, p):
        error = phase_amplitude_damping_error(p, 1 - p)
        self.noise.add_quantum_error(error, instructions=["rz"], qubits=[qubit], warnings=False)
        return 0

    def pha_1q_sx(self, arity, qubit, p):
        error = phase_amplitude_damping_error(p, 1 - p)
        self.noise.add_quantum_error(error, instructions=["sx"], qubits=[qubit], warnings=False)
        return 0

    def init_prob(self,arity=0):
        return random.random()

    def init_qubit(self,arity=0):
        return random.randint(0, 1)

    def fix_prob(self,arity=0):
        return 0.1

    def fix_qubit(self,arity=0):
        return 0


def execute_circuit_v2(circuit, noisemodel):
    sim = AerSimulator(noise_model=noisemodel)
    counts = sim.run(circuit, shots=4096, seed_simulator=42).result().get_counts()
    return counts
