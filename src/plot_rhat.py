from config import *
from shared_utils import *
from plot_utils import *
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec, GridSpec, GridSpecFromSubplotSpec
import gc

name = {
    "campylobacter": "campylob.", 
    "rotavirus": "rotavirus", 
    "borreliosis": "borreliosis"
}

plot_args = {
    "W_ia": {"color": "C1", "label": "$W_{IA}$", "markersize":5},
    "W_t_t": {"color": "C1", "label": "$W_{T_T}$", "markersize":5},
    "W_t_s": {"color": "C1", "label": "$W_{T_S}$", "markersize":5},
    "W_ts": {"color": "C1", "label": "$W_{D}$", "markersize":5},
    "W_s": {"color": "C1", "label": "$W_{E/W}$", "markersize":5},
}

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)

fig = plt.figure(figsize=(12, 7))
grid = GridSpec(1, len(diseases), top=0.9, bottom=0.1, left=0.07, right=0.97, hspace=0.25, wspace=0.15)
    
for i,disease in enumerate(diseases):
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
        
    trace = load_trace(disease, use_age, use_eastwest)
            
    rhatplot(trace, var_names=["W_ia", "W_t_t", "W_t_s", "W_ts", "W_s"], fig=fig, sp=grid[i], bound=1.05, ylabels=(i==0), yticks=False, yticklabels=False, var_args = plot_args, title=name[disease])
    
    bbox = grid[i].get_position(fig)
    fig.text(bbox.x0, bbox.y0+bbox.height+0.005, r"$\textbf{"+str(i+1)+"ABC"[i]+r"}$", fontsize=22)

    del trace
    gc.collect()
    
plt.savefig("../figures/rhat.pdf")
