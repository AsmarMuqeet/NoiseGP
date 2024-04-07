import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from deap import gp as Genetic
from networkx.drawing.nx_agraph import graphviz_layout

if __name__ == '__main__':

    GP_files = sorted([i for i in os.listdir("results/") if "GP" in i],
                      key=lambda x: int(x.split("_")[1].replace(".pkl", "")))
    RP_files = sorted([i for i in os.listdir("results/") if "RP" in i],
                      key=lambda x: int(x.split("_")[1].replace(".pkl", "")))

    #GP_files = [x for x in GP_files if x.replace("GP","RP") in RP_files]
    #print(GP_files,RP_files)
    types = []
    fitness = []
    iteration = []
    noises = []
    for gp,rp in zip(GP_files,RP_files):
        file = open(f"results/{gp}", "rb")
        gp_data = pickle.load(file)
        file.close()
        file = open(f"results/{rp}", "rb")
        rp_data = pickle.load(file)
        file.close()

        for gp_fit,rp_fit,i in zip(gp_data,rp_data,range(10)):
            nodes, edges, labels = Genetic.graph(gp_fit)
            print(labels)
            g = nx.Graph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            pos = graphviz_layout(g, prog="dot")

            nx.draw_networkx_nodes(g, pos)
            nx.draw_networkx_edges(g, pos)
            nx.draw_networkx_labels(g, pos, labels)
            plt.show()
            break

    # with open("models/example_2.pkl","rb") as file:
    #     pop = pickle.load(file)
    #     gp_fit = pop["expression"]
    # nodes, edges, labels = Genetic.graph(gp_fit)
    # inv = {'pha_1q_rx': "ph1RX", 'pha_1q_rz': "ph1RZ", 'ap_1q_sx': "ap1SX", 'ap_1q_ry': "ap1Ry", 'IN0': "I", 1: 50, 0.5846164339917934: 0.58,
    #  0.4006871664170438: 0.40, 0.20214752462876417: 0.20, 'ph_1q_rx': "ph1RX", 'ap_1q_rx': "ap1RX", 0.9461366499687505: 0.94,
    #  'dp_1q_rz': "dp1RZ", 0.23486501379545843: 0.23, 'fix_prob': 1.0, 'ph_1q_rz': "ph1RZ", 0.33864721013946886: 0.33, 'pha_1q_sx': "pa1SX",
    #  'dp_1q_rx': "dp1RX", 0.06374544691307316: 0.06, 0.5313933559861808: 0.53, 'ap_1q_rz': "ap1RZ", 0.14777220059182028: 0.14, 0: 51,
    #  0.7668185687871398: 0.76}
    #
    # #print(labels)
    # for key in labels.keys():
    #     labels[key] = inv[labels[key]]
    # g = nx.Graph()
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # pos = graphviz_layout(g, prog="dot")
    #
    # plt.figure(figsize=(12, 12))
    # nx.draw_networkx_nodes(g, pos)
    # nx.draw_networkx_edges(g, pos)
    # nx.draw_networkx_labels(g, pos, labels)
    # plt.show()