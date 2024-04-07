import pickle
from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from primitives import *
from qiskit.circuit.random import random_circuit
from qiskit import transpile
from primitives_v2 import *
from tqdm import tqdm


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
    fitness = hellinger_distance(counts1, counts2)
    return fitness,


if __name__ == '__main__':

    qc = random_circuit(num_qubits=2,depth=5,measure=True)
    qc = transpile(qc,basis_gates=["sx","id","rx","ry","rz","cx"])

    max_depth = [3,5,7,11,13]
    for depth in max_depth:
        noise = get_random_model(depth)
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
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", calculate_fitness_gp, benchmark=benchmark)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=5)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        pop = toolbox.population(n=1)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        #mstats.register("avg", np.mean)
        #mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=mstats,
                                       halloffame=hof, verbose=True)

        mymodel = benchmark["mynoise"]
        func = toolbox.compile(expr=hof[0])
        mymodel.__reset__()
        func(0)
        result = {"model":mymodel.noise, "expression":str(hof[0])}
        with open(f"models/result_{depth}.pkl", "wb") as outfile:
            pickle.dump(result,outfile)

    for depth in [3,5,7,11,13]:
        with open(f"models/result_{depth}.pkl", "rb") as outfile:
            result = pickle.load(outfile)
            print(result)
            print("------------")