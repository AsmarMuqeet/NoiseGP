import pickle
from functools import partial

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from qiskit import transpile, QuantumCircuit
from primitives_v2 import *
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut
from qiskit_ibm_runtime import QiskitRuntimeService

def run_gp(circuit_file, noise_file,iterations=10):

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
        max_depth = 2
        mymodel = myNoiseModel()
        benchmark = {"circuit":qc,"noise":noise,"mynoise":mymodel}

        pset = gp.PrimitiveSetTyped("MAIN", [int], int, "IN")
        pset.addPrimitive(mymodel.dp_1q_rx,[int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_sx, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_2q_cx, [int, float], int)

        pset.addPrimitive(mymodel.ap_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.ph_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.pha_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.fix_prob,[int],float)
        pset.addPrimitive(mymodel.fix_qubit,[int],int)

        pset.addEphemeralConstant("qubit", partial(mymodel.init_qubit), int)
        pset.addEphemeralConstant("probability", partial(mymodel.init_prob), float)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def calculate_fitness_gp(individual, benchmark):
            qc = benchmark["circuit"]
            noise = benchmark["noise"]
            mymodel = benchmark["mynoise"]
            counts1 = execute_circuit_v2(qc, noise)
            # print(individual)
            func = toolbox.compile(expr=individual)
            mymodel.__reset__()
            func(0)
            counts2 = execute_circuit_v2(qc, mymodel.noise)
            if isinstance(counts1,list):
                avg_fitness = []
                for c1,c2 in zip(counts1,counts2):
                    avg_fitness.append(hellinger_distance(c1,c2))

                fitness = np.mean(avg_fitness)
                return fitness,
            else:
                fitness = hellinger_distance(counts1, counts2)
                return fitness,

        def timed_fitness(indivdual, benchmark):
            try:
                fitness = func_timeout(timeout=30,func=calculate_fitness_gp,args=(indivdual,benchmark))
                return fitness
            except FunctionTimedOut:
                return 1,

        toolbox.register("evaluate", timed_fitness, benchmark=benchmark)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        #mstats.register("avg", np.mean)
        #mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                       halloffame=hof, verbose=True)
        # print log
        #print("log ",log)
        #print("hof",hof[0])
        #print(hof[0].fitness)
        run_results.append(hof[0])

    if isinstance(circuit_file,list):
        file = open(f"results/{noise_file}_GP.pkl", "wb")
        pickle.dump(run_results, file)
        file.close()
    else:
        file = open(f"results/{circuit_file}_{noise_file}_GP.pkl", "wb")
        pickle.dump(run_results,file)
        file.close()

def run_gp_real(circuit_file, noise_file,iterations=10):
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
        max_depth = 2
        mymodel = myNoiseModel()
        benchmark = {"circuit":qc,"real":real,"mynoise":mymodel}

        pset = gp.PrimitiveSetTyped("MAIN", [int], int, "IN")
        pset.addPrimitive(mymodel.dp_1q_rx,[int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_1q_sx, [int, int, float], int)
        pset.addPrimitive(mymodel.dp_2q_cx, [int, float], int)

        pset.addPrimitive(mymodel.ap_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.ap_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.ph_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.ph_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.pha_1q_rx, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_ry, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_rz, [int, int, float], int)
        pset.addPrimitive(mymodel.pha_1q_sx, [int, int, float], int)

        pset.addPrimitive(mymodel.fix_prob,[int],float)
        pset.addPrimitive(mymodel.fix_qubit,[int],int)

        pset.addEphemeralConstant("qubit", partial(mymodel.init_qubit), int)
        pset.addEphemeralConstant("probability", partial(mymodel.init_prob), float)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_depth)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def calculate_fitness_gp(individual, benchmark):
            qc = benchmark["circuit"]
            mymodel = benchmark["mynoise"]
            counts1 = benchmark["real"]
            # print(individual)
            func = toolbox.compile(expr=individual)
            mymodel.__reset__()
            func(0)
            counts2 = execute_circuit_v2(qc, mymodel.noise)
            if isinstance(counts1,list):
                avg_fitness = []
                for c1,c2 in zip(counts1,counts2):
                    avg_fitness.append(hellinger_distance(c1,c2))

                fitness = np.mean(avg_fitness)
                return fitness,
            else:
                fitness = hellinger_distance(counts1, counts2)
                return fitness,

        def timed_fitness(indivdual, benchmark):
            try:
                fitness = func_timeout(timeout=30,func=calculate_fitness_gp,args=(indivdual,benchmark))
                return fitness
            except FunctionTimedOut:
                return 1,

        toolbox.register("evaluate", timed_fitness, benchmark=benchmark)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=max_depth)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        #mstats.register("avg", np.mean)
        #mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                       halloffame=hof, verbose=True)
        # print log
        #print("log ",log)
        #print("hof",hof[0])
        #print(hof[0].fitness)
        run_results.append(hof[0])

    if isinstance(circuit_file,list):
        file = open(f"results/{noise_file}_GP.pkl", "wb")
        pickle.dump(run_results, file)
        file.close()
    else:
        file = open(f"results/{circuit_file}_{noise_file}_GP.pkl", "wb")
        pickle.dump(run_results,file)
        file.close()
