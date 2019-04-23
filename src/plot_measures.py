
import matplotlib
from matplotlib import pyplot as plt
plt.style.use("ggplot")
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pickle as pkl
import numpy as np

with open('../data/measures_summary.pkl',"rb") as f:
  summary = pkl.load(f)

measures = ["deviance", "DS score"]

fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(len(measures),len(summary), top=0.92, bottom=0.07, left=0.1, right=0.98, hspace=0.3, wspace=0.24)
sp = np.empty((len(measures),len(summary)), dtype=object)

for j,measure in enumerate(measures):
    for i,disease in enumerate(summary.keys()):

        sp[j,i] = plt.subplot(grid[j,i])

        if j==0:
            plt.title("campylob." if disease == "campylobacter" else disease, fontsize=22)
        if i==0:
            plt.ylabel(measure+"\n"+" distribution", fontsize=22)

        plt.tick_params(axis="both", direction='out', size=6, labelsize=18)

        for k,model in enumerate(summary[disease].keys()):
            sns.distplot(summary[disease][model][measure].mean(), label=model)

        fig.text(0,1+0.025, r"$\textbf{"+str(i+1)+"ABCDEFGHIJKLMNOPQRSTUVXYZ"[j]+r"}$", fontsize=22, transform=sp[j,i].transAxes)

        if (j,i) == (1,2):
            plt.legend(["hhh4 model", "proposed model"], fontsize=15)

plt.show()
# plt.savefig("../figures/measures.pdf")
