
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from shared_utils import load_data, split_data, plot_counties
import pickle as pkl
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import rc
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

disease = "campylobacter"
prediction_region = "germany"
prediction_week = 30
xlim = (5.5,15.5)
ylim = (47,56)

# load models
with open('../data/results.pkl',"rb") as f:
    models = pkl.load(f)
model = models[disease]["model"]

# load testing dataset
data = load_data(disease, prediction_region, model.counties)
_, _, _, _, data_test, target_test = split_data(data)
# extract county data in the same order as in the model (necessary for plotting)
contrib = model.get_linear_contributions(target_test.index, list(model.counties.keys()), data_test)
interaction = contrib["interaction"]

fig = plt.figure(figsize=(12,12))
fig.suptitle("Contribution from interaction components", fontsize=22)
grid = plt.GridSpec(4, 4)
plt.subplots_adjust(top=0.9, bottom=0.06, left=0.06, right=0.99, hspace=0.3, wspace=0.0)

for j in range(4):
    for i in range(4):
        ax = fig.add_subplot(grid[j,i])
        plot_counties(ax, model.counties, interaction[prediction_week,:,np.ravel_multi_index((i,j),(4,4))], xlim=xlim, ylim=ylim, contourcolor="black", background=(0.8,0.8,0.8,0.8), xticks=False, yticks=False, grid=False, frame=True, ylabel=False, xlabel=False, lw=2)

fig.text(0.5, 0.02, "Longitude", ha='center', fontsize=20)
fig.text(0.08, 0.47, "Latitude", va='center', rotation='vertical', fontsize=20)
plt.show()
