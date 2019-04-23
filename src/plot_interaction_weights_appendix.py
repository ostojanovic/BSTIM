
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("ggplot")
from shared_utils import load_data, split_data, pairplot
from matplotlib import rc
import pymc3 as pm, seaborn as sns
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

diseases = ["campylobacter", "rotavirus", "borreliosis"]

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)

for i,disease in enumerate(diseases):
    fig = plt.figure(figsize=(12,12))
    #grid = plt.GridSpec(1,1,figure=fig, top=0.85, bottom=0.3, left=0.10, right=0.95, hspace=0.4, wspace=0.24)

    axes = []
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
    filename_pred = "../data/mcmc_samples/parameters_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)

    with open(filename_pred,"rb") as f:
        trace = pkl.load(f)
        params = pm.trace_to_dataframe(trace, varnames=["W_ia"])

    labels = ["$\omega_{{{},{}}}$".format(i,j) for i in range(4) for j in range(4)]

    axes = pairplot(params,
        fig=fig, diagonal_kind="kde", lower_kind="kde", upper_kind="empty",
        labels=dict(zip(params.columns,labels)),
        xlabelrotation=90, ylabels=True, ylabelrotation=0,
        tick_args = {"labelbottom": False, "labelleft": False},
        diagonal_kwargs={"color": sns.color_palette()[0]},
        rasterized=False)

    # fig.savefig("../figures/interaction_weights_{}_appendix.pdf".format(disease))
plt.show()
