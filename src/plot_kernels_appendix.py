
import datetime
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from shared_utils import load_data, split_data
from sampling_utils import *
import pickle as pkl
import numpy as np
import matplotlib
from matplotlib import rc
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams["font.family"] = "Bitstream Charter"

from ModelInstance import ModelInstance
diseases = ["campylobacter", "rotavirus", "boreliosis"]
prediction_regions = ["germany", "bavaria"]

locs = np.linspace(-100,100,200)
ts = np.linspace(0*7*24*3600, 5*7*24*3600, 200)

dt = tt.fvector("dt")
dx = tt.fvector("dx")
temporal_bfs = tt.stack(bspline_bfs(dt, np.array([0,0,1,2,3,4,5])*7*24*3600.0, 2),axis=0)
spatial_bfs = tt.stack([gaussian_bf(dx,σ) for σ in [6.25, 12.5, 25.0, 50.0]],axis=0)

temporal_bfs = theano.function([dt], temporal_bfs, allow_input_downcast=True)(ts)
spatial_bfs = theano.function([dx], spatial_bfs, allow_input_downcast=True)(locs)


fig = plt.figure(figsize=(12,12))
grid = plt.GridSpec(len(spatial_bfs)+1,len(temporal_bfs)+1, top=0.92, bottom=0.07, left=0.09, right=0.93, hspace=0.2, wspace=0.2)

ax0 = fig.add_subplot(grid[:,:])
ax00 = fig.add_subplot(grid[0,0])
#ax00.pcolormesh(_t, x, Kc.T)
#ax00.set_xticks(tticks)
#ax00.set_xticklabels(tticks_l)
#ax00.xaxis.tick_top()
ax00.set_visible(False)
#ax00.set_xlim(-1,30)

for k in range(4):
    ax = fig.add_subplot(grid[0, 1+k], sharex=ax00)
    ax.plot(ts/(24*3600),temporal_bfs[k,:])
    ax.set_ylim(0,1)
    ax.xaxis.tick_top()
    ax.tick_params(labelsize=18, length=6, labelleft=k==0)

for k in range(4):
    ax = fig.add_subplot(grid[1+k, 0], sharey=ax00)
    ax.plot(spatial_bfs[k,:],locs)
    ax.set_xlim(0.1,0)
    ax.xaxis.tick_bottom()
    ax.tick_params(labelsize=18, length=6, labelbottom=k==3)

for i in range(4):
    for j in range(4):
        ax = fig.add_subplot(grid[1+i, 1+j], sharex=ax00, sharey=ax00)
        K = spatial_bfs[i,:].reshape((-1,1)) * temporal_bfs[j,:].reshape((1,-1))
        ax.contourf(ts/(24*3600), locs, K)
        ax.set_rasterized(True)
        ax.tick_params(labelbottom=False, labelleft=False, labelsize=18, length=6)


ax0.set_xlabel("temporal distance [days]", fontsize=22)
ax0.set_ylabel("spatial distance [km]", fontsize=22)
ax0.set_frame_on(False)
ax0.set_xticks([])
ax0.set_yticks([])
ax0.yaxis.set_label_position("right")
#fig.suptitle("Interaction Effect Basis", fontsize=18)
# fig.savefig("../figures/kernels_appendix.pdf")
plt.show()
