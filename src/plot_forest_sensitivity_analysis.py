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
matplotlib.rcParams['legend.fontsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['axes.titlesize'] = 22

diseases = ["borreliosis","campylobacter", "rotavirus"]
prediction_regions = ["germany", "bavaria"]

# sample_dir="../data/mcmc_samples"
# figure_dir="../figures/"
# suffix=""
sample_dir="/data/BSTIM/data/sensitivity_analysis/mcmc_samples"
figure_dir="/data/BSTIM/figures/sensitivity_analysis/"
combine=True

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)


for disease in diseases:
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
    prediction_region         = "bavaria" if disease=="borreliosis" else "germany"
    
    vars1 = ["W_ia"]
    vars2 = ["W_t_t", "W_t_s", "W_ts", "W_s"]
    all_vars = vars1 + vars2

    stats_ia = OrderedDict()
    stats_others = OrderedDict()

    for v in vars1:
        stats_ia[v] = {"ids": [], "traces": []}
    for v in vars2:
        stats_others[v] = {"ids": [], "traces": []}


    for scale in prior_scales:
        suffix = "_{:0.3}".format(scale)
        trace = load_trace(disease, use_age, use_eastwest, dir=sample_dir, suffix=suffix)
        tmp_stats_ia = get_trace_stats(trace, combine=True, vars=vars1, include_vars=all_vars)
        tmp_stats_others = get_trace_stats(trace, combine=True, vars=vars2, include_vars=all_vars)
        del trace

        for v in vars1:
            if v in tmp_stats_ia:
                stats_ia[v]["ids"] = tmp_stats_ia[v]["ids"]
                stats_ia[v]["traces"].append({"quantiles": tmp_stats_ia[v]["quantiles"], "means": tmp_stats_ia[v]["means"]})
        for v in vars2:
            if v in tmp_stats_others:
                stats_others[v]["ids"] = tmp_stats_others[v]["ids"]
                stats_others[v]["traces"].append({"quantiles": tmp_stats_others[v]["quantiles"], "means": tmp_stats_others[v]["means"]})

    fig = plt.figure(figsize=(12, 14))
    grid = GridSpec(1, 2, top=0.9, bottom=0.1, left=0.07, right=0.97, hspace=0.25, wspace=0.15)

    W_ia_args = {"W_ia": {"color": ["C1","C2","C3","C4","C5"], "label": "$W_{IA}$", "markersize":4, "interquartile_linewidth": 2, "credible_linewidth": 1}}
    other_args = {
        "W_t_t": {"color": ["C1","C2","C3","C4","C5"], "label": "$W_{T_T}$"},
        "W_t_s": {"color": ["C1","C2","C3","C4","C5"], "label": "$W_{T_S}$"},
        "W_ts": {"color": ["C1","C2","C3","C4","C5"], "label": "$W_{D}$"},
        "W_s": {"color": ["C1","C2","C3","C4","C5"], "label": "$W_{E/W}$"},
    }
    forestplot(stats_ia, group_args=W_ia_args, fig=fig, sp=grid[0], combine=combine)
    forestplot(stats_others, group_args=other_args, fig=fig, sp=grid[1], combine=combine)
    plt.figlegend([plt.Line2D([],[],linewidth=2, marker="o", markersize=4, color=c) for c in ["C1","C2","C3","C4","C5"]], ["{}".format(scale) for scale in prior_scales], loc="center", bbox_to_anchor=(0.5,0.05), ncol=5)
    
    
    gc.collect()
    plt.savefig(os.path.join(figure_dir,"forest_{}_sensitivity.pdf".format(disease)))

    plt.close()
    
