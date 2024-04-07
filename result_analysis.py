import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    sns.boxplot(data=data,x="noise",y="fitness",hue="type")
    plt.show()
    groups = data.groupby(["noise","type"])
    groups.describe()[["fitness"]].T.to_csv("fitness.csv")



    #
    # with open("results/result_11.pkl_GP.pkl","rb") as file:
    #     data = pickle.load(file)
    #
    # print(data[0])
    # print([x.fitness for x in data])
    #
    # with open("results/result_11.pkl_RP.pkl","rb") as file:
    #     data = pickle.load(file)
    #
    # print(data)