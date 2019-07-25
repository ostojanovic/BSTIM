from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime, pickle as pkl, numpy as np, matplotlib, pandas as pd, pymc3 as pm
from pymc3.stats import quantiles
from config import *
from plot_utils import *
from shared_utils import *
from sampling_utils import *
from matplotlib import rc
import isoweek
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from BaseModel import BaseModel
import gc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['axes.titlesize'] = 22

diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)
    
for disease in diseases:
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
    prediction_region         = "bavaria" if disease=="borreliosis" else "germany"
    filename_params = "../data/mcmc_samples/parameters_{}_{}_{}".format(disease, use_age, use_eastwest)
    filename_model = "../data/mcmc_samples/model_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)

    with open(filename_model,"rb") as f:
        model = pkl.load(f)

    with model:
        trace = pm.load_trace(filename_params)

    fig = plt.figure(figsize=(12, 14))
    grid = GridSpec(1, 2, top=0.9, bottom=0.1, left=0.07, right=0.97, hspace=0.25, wspace=0.15)

    W_ia_args = {"W_ia": {"color": "C1", "label": "$W_{IA}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1}}
    other_args = {
        "W_t_t": {"color": "C1", "label": "$W_{T_T}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1},
        "W_t_s": {"color": "C1", "label": "$W_{T_S}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1},
        "W_ts": {"color": "C1", "label": "$W_{D}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1},
        "W_s": {"color": "C1", "label": "$W_{E/W}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1},
    }

    ia_range = range(1,17)
    t_t_range = range(17,22)
    t_s_range = range(22,26)
    ts_range = range(26,29)
    s_range = range(29,30)

    forestplot(trace, var_labels=[("W_ia", ia_range)], var_args=W_ia_args, fig=fig, sp=grid[0])
    forestplot(trace, var_labels=[("W_t_t", t_t_range), ("W_t_s", t_s_range), ("W_ts", ts_range), ("W_s", s_range)], var_args=other_args, fig=fig, sp=grid[1])
    plt.savefig("../figures/forest_{}.pdf".format(disease))

    del model
    del trace
    gc.collect()
    
