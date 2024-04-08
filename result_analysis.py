import os
import pickle

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
r = robjects.r
r.library('effsize')
r['source']('A12.R')
test = robjects.globalenv['A12']


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
            fitness.append(gp_fit.fitness.values[0])
            types.append("GP")
            iteration.append(i)
            noises.append(gp.replace(".pkl_GP.pkl",""))

            fitness.append(rp_fit)
            types.append("RP")
            iteration.append(i)
            noises.append(rp.replace(".pkl_RP.pkl",""))

    data = pd.DataFrame()
    data["noise"] = noises
    data["type"] = types
    data["fitness"] = fitness
    data["iteration"] = iteration

    #sns.boxplot(data=data,x="noise",y="fitness",hue="type")
    #plt.show()
    groups = data.groupby(["noise","type"])
    groups.describe()[["fitness"]].T.to_csv("fitness.csv")

    noises = np.unique(data["noise"])
    pvalues = []
    effectsizes = []
    magnitudes = []
    for noise in noises:
        fitness_gp = data[(data["noise"]==noise) & (data["type"]=="GP")]["fitness"].values
        fitness_rp = data[(data["noise"] == noise) & (data["type"] == "RP")]["fitness"].values
        st = scipy.stats.mannwhitneyu(fitness_gp,fitness_rp)
        pvalue = np.round(st.pvalue,4)
        df_result_r = test(robjects.FloatVector(fitness_gp),
                           robjects.FloatVector(fitness_rp))

        mag = str(df_result_r[2]).split("\n")[0].split()[-1]
        oeffect = float(str(df_result_r[3]).split()[-1])
        pvalues.append(pvalue)
        magnitudes.append(mag)
        effectsizes.append(oeffect)

    statistics = pd.DataFrame({"noise":noises,"pvalue":pvalues,"effectsize":effectsizes, "magnitude":magnitudes})
    statistics.to_csv("statistics.csv",index=False)