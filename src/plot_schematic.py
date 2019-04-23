import matplotlib.pyplot as plt
plt.style.use('ggplot')
from shared_utils import load_data, split_data, plot_counties, make_axes_stack
import pickle as pkl
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import rc
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amssymb}"]
plt.rcParams["font.family"] = "Bitstream Charter"

from BaseModel import BaseModel
from collections import OrderedDict
from pandas import DataFrame

disease = "campylobacter"
prediction_region = "germany"
prediction_week = 30
xlim = (5.5,15.5)
ylim = (47,56)

    
# load models
with open('../data/counties/counties.pkl',"rb") as f:
    counties = pkl.load(f)
    

# load data
data = load_data(disease, prediction_region, counties)
data_train, target_train, data_test, target_test = split_data(data)

with open("../data/ia_effect_samples/{}_{}.pkl".format(disease, 0), "rb") as f:
    ia_effects = pkl.load(f)
ia_effects = DataFrame(ia_effects["ia_effects"][812,:,:], index=data.columns)
    
# recreate model
model = BaseModel((target_train.index[0],target_train.index[-1]), counties, ["../data/ia_effect_samples/{}_{}.pkl".format(disease, i) for i in range(100)], include_eastwest=True, include_demographics=True)

# evaluate features
f_all = model.evaluate_features(data.index[812:813], data.columns) 

# create dictionaries for individual contributions
contributions = OrderedDict()
contributions["interaction"] = [OrderedDict(ia_effects[i]) for i in range(16)]
contributions["temporal"]    = [OrderedDict(f_all["temporal_seasonal"].loc[(2016,30)][c].to_dict()) for c in f_all["temporal_seasonal"].columns] + [OrderedDict(f_all["temporal_trend"][c].to_dict()) for c in f_all["temporal_trend"].columns]
contributions["political"]   = [OrderedDict(f_all["spatial"].loc[(2016,30)][c].to_dict()) for c in f_all["spatial"].columns]
contributions["demographical"] = [OrderedDict(f_all["spatiotemporal"].loc[(2016,30)][c].to_dict()) for c in f_all["spatiotemporal"].columns]
contributions["exposure"]    = [OrderedDict(f_all["exposure"].loc[(2016,30)][c].to_dict()) for c in f_all["exposure"].columns]



fig = plt.figure(figsize=(12,4))
fig.subplots_adjust(wspace=0.5, hspace=0.5)
# grid = plt.GridSpec(2, 5, top=0.65, bottom=0.1, left=0.15, right=0.95, hspace=0.33, wspace=0.5)
sum_row=False
grid = plt.GridSpec(1+sum_row+4, 5, top=0.75, bottom=0.1, left=0.12, right=0.95, hspace=0.33, wspace=0.5, height_ratios=[1]+([1] if sum_row else [])+[0.1, 0.1, 0.1, 0.1,])

ratio = fig.get_figwidth()/fig.get_figheight()

in_models = {"interaction": 3, "temporal": 3, "political": 2, "demographical": 1, "exposure": 3}

axes=[[],[]]
for i,(name,contrib) in enumerate(contributions.items()):
    axes[0].append(make_axes_stack(grid[0,i], len(contrib), 0.01/ratio, 0.015, down=True))
    for j in range(len(contrib)):
        plot_counties(axes[0][i][j], counties, contrib[j], xlim=xlim, ylim=ylim, contourcolor="black", background=(0.8,0.8,0.8,0.8), xticks=False, yticks=False, grid=False, frame=True, ylabel=False, xlabel=False, lw=2)
        axes[0][i][j].set_rasterized(True)
    
    axes[0][i][0].set_xlabel(name, fontsize=22)

    for j,m in enumerate("ABC"):
        fig.text(*grid[1+sum_row+j+1, i].get_position(fig).corners().mean(axis=0), r"{\LARGE"+(r"\checkmark" if j >= 3-in_models[name] else r"\textbf{\--}")+r"}", fontsize=30, verticalalignment="center", horizontalalignment="center")
        

for i in range(3):
    bbox = grid[1+sum_row+i+1,0].get_position(fig)
    fig.text(bbox.x0, (bbox.y0+bbox.y1)/2, r"{\LARGE model \textbf{"+"ABC"[i]+r"}:}", fontsize=30, verticalalignment="center", horizontalalignment="right")
    
# write "exp("
fig.text(axes[0][0][0].get_position().x0, (axes[0][0][0].get_position().y0+axes[0][0][0].get_position().y1)/2, r"exp{$\Huge[$}", fontsize=40, verticalalignment="center", horizontalalignment="right", zorder=100)
# write all "+"
for i in range(3):
    fig.text((axes[0][i][0].get_position().x1+axes[0][i+1][0].get_position().x0)/2, (axes[0][i][0].get_position().y0+axes[0][i][0].get_position().y1)/2, r"$\textbf{+}$", fontsize=40, verticalalignment="center", horizontalalignment="center", zorder=90, color="white")
    fig.text((axes[0][i][0].get_position().x1+axes[0][i+1][0].get_position().x0)/2, (axes[0][i][0].get_position().y0+axes[0][i][0].get_position().y1)/2, r"$+$", fontsize=40, verticalalignment="center", horizontalalignment="center", zorder=100)
# write ")⋅"
fig.text((axes[0][-2][0].get_position().x1+axes[0][-1][0].get_position().x0)/2, (axes[0][-1][0].get_position().y0+axes[0][-1][0].get_position().y1)/2, r"{$\Huge]$}$\cdot$", fontsize=40, verticalalignment="center", horizontalalignment="center", zorder=100)

fig.savefig("../figures/schematic.pdf")
