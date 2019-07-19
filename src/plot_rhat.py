from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime, pickle as pkl, numpy as np, matplotlib, pandas as pd, pymc3 as pm
from pymc3.stats import quantiles
from shared_utils import load_data, split_data, forestplot, rhatplot
from sampling_utils import *
from matplotlib import rc
import isoweek
import gc
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from BaseModel import BaseModel
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['axes.titlesize'] = 22

diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

name = {"campylobacter": "campylob.", "rotavirus": "rotavirus", "borreliosis": "borreliosis"}


fig = plt.figure(figsize=(12, 7))
grid = GridSpec(1, len(diseases), top=0.9, bottom=0.1, left=0.07, right=0.97, hspace=0.25, wspace=0.15)

plot_args = {
    "W_ia": {"color": "C1", "label": "$W_{IA}$", "markersize":5},
    "W_t_t": {"color": "C1", "label": "$W_{T_T}$", "markersize":5},
    "W_t_s": {"color": "C1", "label": "$W_{T_S}$", "markersize":5},
    "W_ts": {"color": "C1", "label": "$W_{D}$", "markersize":5},
    "W_s": {"color": "C1", "label": "$W_{E/W}$", "markersize":5},
}
    
for i,disease in enumerate(diseases):
    use_age = True#best_model[disease]["use_age"]
    use_eastwest = True#best_model[disease]["use_eastwest"]
    prediction_region = "bavaria" if disease=="borreliosis" else "germany"
    filename_params = "../data/mcmc_samples/parameters_{}_{}_{}".format(disease, use_age, use_eastwest)
    filename_model = "../data/mcmc_samples/model_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)

    with open(filename_model,"rb") as f:
        model = pkl.load(f)

    with model:
        trace = pm.load_trace(filename_params)
            
    rhatplot(trace, var_names=["W_ia", "W_t_t", "W_t_s", "W_ts", "W_s"], fig=fig, sp=grid[i], bound=1.05, ylabels=(i==0), yticks=False, yticklabels=False, var_args = plot_args, title=name[disease])
    
    bbox = grid[i].get_position(fig)
    fig.text(bbox.x0, bbox.y0+bbox.height+0.005, r"$\textbf{"+str(i+1)+"ABC"[i]+r"}$", fontsize=22)
    
    del model
    del trace
    gc.collect()
    
plt.savefig("../figures/rhat.svg")
