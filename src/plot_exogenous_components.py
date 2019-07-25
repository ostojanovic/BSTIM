from config import *
from shared_utils import *
from plot_utils import *
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("ggplot")
from matplotlib import rc
import pymc3 as pm, seaborn as sns

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)

# refrence values from the hhh4 model
reference_alpha = {"campylobacter": 0.1626016, "rotavirus":0.5647084, "borreliosis": 0.9351244}

fig = plt.figure(figsize=(12,8))
grid = plt.GridSpec(2, len(diseases), figure=fig, top=0.93, bottom=0.09, left=0.12, right=0.95, hspace=0.65, wspace=0.24, height_ratios=[2,1])

for i,disease in enumerate(diseases):
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]

    trace = load_trace(disease, use_age, use_eastwest)
    params_categorical = pm.trace_to_dataframe(trace, varnames=["W_ts","W_s"])
    params_alpha = pm.trace_to_dataframe(trace, varnames=["α"])

    if "W_s__0" not in params_categorical.columns:
        params_categorical["W_s__0"] = np.nan

    # Plot demographic and political parameter distributions
    axes = pairplot(params_categorical,
        fig=fig, spec=grid[0,i], diagonal_kind="kde", lower_kind="kde", upper_kind="empty",
        labels={"W_ts__0": "[0-5)", "W_ts__1": "[5-20)", "W_ts__2": "[20-65)", "W_s__0": "east/west"},
        xlabelrotation=90, ylabels=(i==0), ylabelrotation=0,
        diagonal_kwargs={"color": sns.color_palette()[0]})

    box=grid[0,i].get_position(fig)
    fig.text(box.xmin, box.ymax, r"$\textbf{"+str(i+1)+"A"+r"}$", fontsize=22, ha="left", va="bottom")
    fig.text((box.xmin+box.xmax)/2, box.ymax, r"{}".format("campylob." if disease == "campylobacter" else disease), fontsize=22, ha="center", va="bottom")

    # Plot alpha parameters
    ax = plt.Subplot(fig, grid[1,i])
    sns.kdeplot(1.0/params_alpha["α"], shade=True, ax=ax,legend=False, color=sns.color_palette()[0])
    ax.axvline(reference_alpha[disease], color="black", ls="dashed", lw=2)
    ax.annotate("hhh4 reference", (reference_alpha[disease],0), textcoords='offset points', xytext=(2,2), fontsize=20, rotation=90, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel(r"$\alpha$", fontsize=22)
    fig.add_subplot(ax)
    ax.tick_params(axis="x", labelsize=18, length=6)

    if disease == "borreliosis":
        ax.set_xlim([0.6, 0.97])

    box=grid[1,i].get_position(fig)
    fig.text(box.xmin, box.ymax, r"$\textbf{"+str(i+1)+"B"+r"}$", fontsize=22, ha="left", va="bottom")

# plt.show()
fig.savefig("../figures/exogenous_components.pdf")
