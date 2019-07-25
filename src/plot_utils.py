import matplotlib
from matplotlib.collections import PatchCollection
import matplotlib.cm
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib import rc
from shapely.geometry import Polygon
from collections import OrderedDict, defaultdict
import numpy as np
from shapely.ops import cascaded_union
from descartes import PolygonPatch
import matplotlib.patheffects as PathEffects
from matplotlib.gridspec import SubplotSpec, GridSpec, GridSpecFromSubplotSpec
import pymc3 as pm
import seaborn as sns
from itertools import product
import re
plt.style.use('ggplot')

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams["font.family"] = "Bitstream Charter"

matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['axes.titlesize'] = 22

def plot_counties(ax, counties, values=None, edgecolors=None, contourcolor="white", hatch_surround=None, xlim=None, ylim=None, background=True, xticks=True, yticks=True, grid=True, frame=True, xlabel="Longitude [in dec. degrees]", ylabel="Latitude [in dec. degrees]", lw=1):
    polygons = [r["shape"] for r in counties.values()]
    # extend german borders :S and then shrink them again after unifying
    # gets rid of many holes at the county boundaries
    contour = cascaded_union([pol.buffer(0.01) for pol in polygons]).buffer(-0.01)

    xmin,ymin,xmax,ymax = contour.bounds
    if xlim is None:
        xlim = [xmin, xmax]
    if ylim is None:
        ylim = [ymin, ymax]

    surround = PolygonPatch(Polygon([(xlim[0],ylim[0]),(xlim[0],ylim[1]),(xlim[1],ylim[1]),(xlim[1],ylim[0])]).difference(contour))
    contour= PolygonPatch(contour, lw=lw)

    pc = PatchCollection([PolygonPatch(p, lw=lw) for p in polygons], cmap=matplotlib.cm.viridis, alpha=1.0)

    if values is not None:
        if isinstance(values,(dict, OrderedDict)):
            values = np.array([values.setdefault(r, np.nan) for r in counties.keys()])
        elif isinstance(values, str):
            values = np.array([r.setdefault(values, np.nan) for r in counties.values()])
        else:
            assert np.size(values) == len(counties), "Number of values ({}) doesn't match number of counties ({})!".format(np.size(values), len(counties))
        #pc.set_clim(np.min(values), np.max(values))
        nans = np.isnan(values)
        values[nans] = 0

        values = np.ma.MaskedArray(values, mask=nans)
        pc.set(array=values, cmap='viridis')
    else:
        pc.set_facecolors("none")

    if edgecolors is not None:
        if isinstance(edgecolors,(dict, OrderedDict)):
            edgecolors = np.array([edgecolors.setdefault(r,"none") for r in counties.keys()])
        elif isinstance(edgecolors, str):
            edgecolors = np.array([r.setdefault(edgecolors, "none") for r in counties.values()])
        pc.set_edgecolors(edgecolors)
    else:
        pc.set_edgecolors("none")

    if hatch_surround is not None:
        surround.set_hatch(hatch_surround)
        surround.set_facecolor("none")
        ax.add_patch(surround)

    ax.add_collection(pc)

    if contourcolor is not None:
        contour.set_edgecolor(contourcolor)
        contour.set_facecolor("none")
        ax.add_patch(contour)

    if isinstance(background, bool):
        ax.patch.set_visible(background)
    else:
        ax.patch.set_color(background)

    ax.grid(grid)

    ax.set_frame_on(frame)


    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1.43)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)

    ax.tick_params(axis="x", which="both", bottom=xticks, labelbottom=xticks)
    ax.tick_params(axis="y", which="both", left=yticks, labelleft=yticks)

    return pc, contour, surround


def pairplot(df, labels={}, diagonal_kind="kde", lower_kind="kde", upper_kind="empty", spec=GridSpec(1,1)[0],
xlabelrotation=0, ylabelrotation=90, ylabels=True, xlabels=True, xtickrotation=60,
fig=plt.gcf(), lower_kwargs={}, diagonal_kwargs={}, upper_kwargs={}, rasterized=False, tick_args={}):
    N = len(df.columns)
    axes = np.empty((N,N),dtype=object)

    g = GridSpecFromSubplotSpec(N,N, subplot_spec=spec)
    fake_axes = {}
    for y in range(N):
        fake_axes[(y,0)] = plt.Subplot(fig, g[y,0])
        fake_axes[(y,0)].set_visible(False)
    for x in range(1,N):
        fake_axes[(0,x)] = plt.Subplot(fig, g[0,x])
        fake_axes[(0,x)].set_visible(False)

    for y,v2 in enumerate(df.columns):
        for x,v1 in enumerate(df.columns):
            if np.all(np.isnan(df[v1])) or np.all(np.isnan(df[v2])):
                axes[y,x] = plt.Subplot(fig, g[y,x], **share_args)
                kind = "noframe"
            else:    
                if y<x: # upper triangle
                    kind = upper_kind
                    kwargs = upper_kwargs
                elif y==x: # diagonal
                    kind = diagonal_kind
                    kwargs = diagonal_kwargs
                else: #lower triangle
                    kind = lower_kind
                    kwargs = lower_kwargs

                if x==y and kind == "kde":
                    share_args={"sharex": fake_axes[(0,x)]}
                    tick_args_default={"left": False, "labelleft": False, "bottom": (y==N-1), "labelbottom": (y==N-1), "labelsize": 18, "length": 6}
                else:
                    share_args={"sharex": fake_axes[(0,x)], "sharey": fake_axes[(y,0)]}
                    tick_args_default={"labelleft": (x==0), "labelright": (x==N-1), "labelbottom": (y==N-1), "left": (x==0), "right": (x==N-1), "bottom": (y==N-1), "labelsize": 18, "length": 6}
                tick_args_default.update(tick_args)
                tick_args = tick_args_default

                axes[y,x] = plt.Subplot(fig, g[y,x], **share_args)
                axes[y,x].tick_params(axis="x", labelrotation=xtickrotation, **tick_args)

            if kind == "noframe":
                axes[y,x].set_frame_on(False)
                axes[y,x].set_xticks([])
                axes[y,x].set_yticks([])
            elif kind == "empty":
                axes[y,x].set_visible(False)
            elif kind == "scatter":
                axes[y,x].scatter(df[v1],df[v2],**kwargs)
            elif kind == "reg":
                sns.regplot(df[v1],df[v2],ax=axes[y,x],**kwargs)
            elif kind == "kde":
                if x==y:
                    sns.kdeplot(df[v1],shade=True, ax=axes[y,x],legend=False, **kwargs)
                    axes[y,x].grid(False)
                else:
                    sns.kdeplot(df[v1],df[v2], shade=True, shade_lowest=False, ax=axes[y,x],legend=False,**kwargs)
                #kde
            else:
                raise NotImplementedError("Subplot kind must be 'empty', 'scatter', 'reg' or 'kde'.")

            axes[y,x].set_rasterized(rasterized)
            if x==0 and ylabels:
                axes[y,x].set_ylabel(labels.setdefault(v2, v2), rotation=ylabelrotation, ha='right', va="center", fontsize=18)
                axes[y,x].tick_params(**tick_args)
            else:
                axes[y,x].set_ylabel("")
                axes[y,x].tick_params(**tick_args)

            if y==N-1 and xlabels:
                axes[y,x].set_xlabel(labels.setdefault(v1, v1), rotation=xlabelrotation, ha='center', va="top", fontsize=18)
            else:
                axes[y,x].set_xlabel("")

            fig.add_subplot(axes[y,x])


    positive = np.all(df.values >= 0)

    for y in range(N):
        if np.all(np.isnan(df.iloc[:,y])):
            continue
        μ = df.iloc[:,y].mean()
        σ = df.iloc[:,y].std()
        if positive:
            fake_axes[(y,0)].set_yticks((0,μ,μ+3*σ))
            fake_axes[(y,0)].set_ylim((0,μ+4*σ))
        else:
            fake_axes[(y,0)].set_yticks((μ-3*σ,μ,μ+3*σ))
            fake_axes[(y,0)].set_ylim((μ-4*σ,μ+4*σ))
        fake_axes[(y,0)].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    for x in range(N):
        if np.all(np.isnan(df.iloc[:,y])):
            continue
        μ = df.iloc[:,x].mean()
        σ = df.iloc[:,x].std()
        if positive:
            fake_axes[(0,x)].set_xticks((0,μ,μ+3*σ))
            fake_axes[(0,x)].set_xlim((0,μ+4*σ))
        else:
            fake_axes[(0,x)].set_xticks((μ-3*σ,μ,μ+3*σ))
            fake_axes[(0,x)].set_xlim((μ-4*σ,μ+4*σ))
        fake_axes[(0,x)].xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    return np.array(axes)

def rhatplot(trace, var_names=None, var_args={}, fig=plt.gcf(), sp=GridSpec(1,1)[:,:], bound=None, ylabels=True, yticks=True, yticklabels=True, title="$\hat R$", labelsize=22):
    if var_names == None:
        var_names = trace.varnames
    var_args = defaultdict(lambda: {"color": "C1", "label": None, "markersize":1}, **var_args)
    num_groups = len(var_names)
    tp = trace.point(0)

    rhat = pm.gelman_rubin(trace, varnames=var_names)

    minval = np.min([np.min(rhat[name]) for name in var_names if len(rhat[name])>0])
    maxval = np.max([np.max(rhat[name]) for name in var_names if len(rhat[name])>0])
    if bound == None:
        bound = maxval
    
    bound_label = str(bound)
    gl,gz,gt = re.match(r"([0-9]+\.)(0*)(.*)", bound_label).groups()
    gt = str(round(int(gt)/10**(len(gt)-1)))[0]
    bound_label = gl + gz + gt
    
    grid = GridSpecFromSubplotSpec(num_groups,1,sp, height_ratios=[np.prod(tp[name].shape)+2 for name in var_names])
    axes = []
    for j,name in enumerate(var_names):
        if len(tp[name])==0:
            continue
            
        ax = fig.add_subplot(grid[j], sharex=axes[0] if len(axes)>0 else None)
        args= var_args[name]
        
        yticks_ = []
        yticklabels_ = []
        for i,idx in enumerate(product(*(range(s) for s in tp[name].shape))):
            yticks_.append(-i)
            yticklabels_.append("{}".format(np.squeeze(idx)))

        if name in rhat:
            ax.plot(rhat[name], yticks_, "o", markersize=args["markersize"])
        
        ax.set_ylim([yticks_[-1]-1, 1])
        
        if yticklabels==False:
            ax.set_yticklabels([])
        elif yticklabels==True:
            ax.set_yticklabels(yticklabels_)
        else:
            ax.set_yticklabels(yticklabels)
            
        if yticks==False:
            ax.set_yticks([])
        elif yticks==True:
            ax.set_yticks(yticks_)
        else:
            ax.set_yticks(yticks)
            
        if ylabels != False:
            bbox = ax.get_position()
            if ylabels==True:
                label = args["label"]
            else:
                label = ylabels[j]
            
            if label == None:
                label = name
                
            fig.text(bbox.x0-0.01, bbox.y0 + bbox.height/2, label, ha="right", va="center", fontsize=labelsize)
        
            
        # ax.set_ylabel(label, rotation=0)
        axes.append(ax)
    axes[-1].set_xticks([1.0, bound])
    axes[-1].set_xticklabels(["1.0", bound_label])
    axes[-1].set_xlim([min(minval,1.0)-0.01, max(bound, maxval)+0.01])
        
    for ax in axes[:-1]:
        for tick in ax.get_xticklabels():
            tick.set_visible(False)
        
    axes[0].set_title(title)
    return axes, grid

# because the trace loading doesnt load energy stats properly...
def energyplot(energies, fill_color=("C0","C1"), fill_alpha=(1,0.5), fig=plt.gcf(), sp=GridSpec(1,1)[:,:]):
    
    for i,energy in enumerate(energies):
        mean_energy, trans_energy = energy - energy.mean(), np.diff(energy)
        ax = fig.add_subplot(sp)
        pm.kdeplot(mean_energy, label="Marginal Energy", ax=ax, shade=fill_alpha[0], kwargs_shade={"color": fill_color[0]})
        pm.kdeplot(trans_energy, label="Energy Transition", ax=ax, shade=fill_alpha[1], kwargs_shade={"color": fill_color[1]})
    
        ax.plot([], label="chain {:>2} BFMI = {:.2f}".format(i, pm.bfmi({"energy":energy})), alpha=0)
    ax.legend()
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    
# because the default forest plot is not flexible enough #sad
def forestplot(trace, var_labels=None, var_args={}, fig=plt.gcf(), sp=GridSpec(1,1)[:,:], combine=False, credible_interval=0.95):
    if var_labels == None:
        var_labels = trace.varnames
    
    var_args = defaultdict(lambda: {"color": "C1", "label": None, "interquartile_linewidth": 2, "credible_linewidth": 1}, **var_args)
    
    num_groups = len(var_labels)
    tp = trace.point(0)
    
    # create indices
    for i,var_label in enumerate(var_labels):
        name = var_label if isinstance(var_label, str) else var_label[0]
        
        cart = product(*(range(s) for s in tp[name].shape))
        if isinstance(var_label, str):
            var_labels[i] = (var_label, map(np.squeeze,cart), cart)
        else:
            var_labels[i] = tuple(var_label) + (cart,)

    def plot_var_trace(ax, y, var_trace, credible_interval=0.95, **args):
        endpoint = (1 - credible_interval) / 2
        qs = np.quantile(var_trace, [endpoint, 1.0-endpoint, 0.25, 0.75])
        ax.plot(qs[:2],[y, y], color=args["color"], linewidth=args["credible_linewidth"])
        ax.plot(qs[2:],[y, y], color=args["color"], linewidth=args["interquartile_linewidth"])
        ax.plot([np.mean(var_trace)], [y], "o", color=args["color"], markersize=args["markersize"])

    grid = GridSpecFromSubplotSpec(num_groups,1,sp, height_ratios=[np.prod(tp[name].shape)+2 for (name,idxs,carts) in var_labels])
    axes = []
    for j,(name,idxs,carts) in enumerate(var_labels):
        if len(tp[name])==0:
            continue
            
        ax = fig.add_subplot(grid[j])
        args= var_args[name]

        yticks = []
        yticklabels = []
        # plot label
        # plot variable stats
        for i,(idx,cart) in enumerate(zip(idxs,carts)):
            yticks.append(-i)
            yticklabels.append("{}".format(idx))
            if combine:
                var_trace = trace[name][(slice(-1),)+cart]
                plot_var_trace(ax, -i, var_trace, credible_interval=credible_interval, **args)
            else:
                for c,chain in enumerate(trace.chains):
                    var_trace = trace.get_values(name, chains=chain)[(slice(-1),)+cart]
                    plot_var_trace(ax, -i+0.25-c/(trace.nchains-1) * 0.5, var_trace, credible_interval=credible_interval, **args)


        ax.set_yticks(yticks)
        ax.set_ylim([yticks[-1]-1, 1])
        ax.set_yticklabels(yticklabels)

        label = args["label"]
        if label == None:
            label = name
        ax.set_ylabel(label)

        # ax.set_frame_on(False)
        axes.append(ax)
    return axes, grid
