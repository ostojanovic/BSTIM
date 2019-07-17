from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime, pickle as pkl, numpy as np, matplotlib, pandas as pd, pymc3 as pm
from pymc3.stats import quantiles
from shared_utils import load_data, split_data, forestplot
from sampling_utils import *
from matplotlib import rc
import isoweek
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from BaseModel import BaseModel
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"
diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

disease = "campylobacter"
use_age = True#best_model[disease]["use_age"]
use_eastwest = True#best_model[disease]["use_eastwest"]
prediction_region         = "bavaria" if disease=="borreliosis" else "germany"
filename_pred = "../data/mcmc_samples/parameters_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)

# with open(filename_pred,"rb") as f:
#     trace = pkl.load(f)

import arviz as az
non_centered_data = az.load_arviz_data('non_centered_eight')

fig = plt.figure()

W_ia_args = {"W_ia": {"color": "C1", "label": "$W_{IA}$", "interquartile_linewidth": 2, "credible_linewidth": 1}}
other_args = {
    "W_t_t": {"color": "C1", "label": "$W_{T_T}$", "interquartile_linewidth": 2, "credible_linewidth": 1},
    "W_t_s": {"color": "C1", "label": "$W_{T_S}$", "interquartile_linewidth": 2, "credible_linewidth": 1},
    "W_ts": {"color": "C1", "label": "$W_{TS}$", "interquartile_linewidth": 2, "credible_linewidth": 1},
    "W_s": {"color": "C1", "label": "$W_{S}$", "interquartile_linewidth": 2, "credible_linewidth": 1},
}

grid = GridSpec(1,2,figure=fig)
forestplot(trace, var_names=["W_ia"], var_args=W_ia_args, fig=fig, sp=grid[0])
forestplot(trace, var_names=["W_t_t", "W_t_s", "W_ts", "W_s"], var_args=other_args, fig=fig, sp=grid[1])