import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from collections import OrderedDict
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from descartes import PolygonPatch
import seaborn as sns
from matplotlib.collections import PatchCollection
import matplotlib.cm
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import re, isoweek
import pymc3 as pm
import theano.tensor as tt
import scipy.stats
import os

yearweek_regex = re.compile(r"([0-9]+)-KW([0-9]+)")
def _parse_yearweek(yearweek):
    """Utility function to convert internal string representations of calender weeks into datetime objects. Uses strings of format `<year>-KW<week>`. Weeks are 1-based."""
    year,week = yearweek_regex.search(yearweek).groups()
    return isoweek.Week(int(year), int(week)) #datetime.combine(isoweek.Week(int(year), int(week)).wednesday(),time(0))
parse_yearweek = np.frompyfunc(_parse_yearweek, 1, 1)

def load_data(disease, prediction_region, counties, separator=";"):
    """
        load_data(disease, prediction_region, counties, separator)

    Utility to load the data for a given disease and prediction_region. Only counties are selected that lie in the prediction_region.

    Arguments:
    ==========
        disease:            name of the disease
        prediction_region:  name of the region for which to make a prediction
        counties:           dictionary containing county information
        separator:          string delimiter
        """
    # load raw data from csv file into a pandas dataframe
    data = pd.read_csv("../data/diseases/{}.csv".format(disease),sep=separator,encoding='iso-8859-1',index_col=0)
    # exclude data reported for unknown counties
    if "99999" in data.columns:
        data.drop("99999", inplace=True, axis=1)
    # exclude data for regions outside of region of interest
    data = data.loc[:, list(filter(lambda cid: prediction_region in counties[cid]["region"], data.columns))]
    data.index = parse_yearweek(data.index)
    return data

def split_data(
            data,
            train_start  = parse_yearweek("2011-KW01"),
            test_start = parse_yearweek("2016-KW01"),
            post_test = parse_yearweek("2018-KW01")
        ):
        """
            split_data(data,data_start,train_start,test_start)

        Utility function that splits the dataset into training and testing data as well as the corresponding target values.

        Returns:
        ========
            data_train:     training data (from beginning of records to end of training phase)
            target_train:   target values for training data
            data_test:      testing data (from beginning of records to end of testing phase = end of records)
            target_test:    target values for testing data
        """

        target_train    = data.loc[(train_start <= data.index)  & (data.index < test_start)]
        target_test     = data.loc[(test_start <= data.index) & (data.index < post_test)]

        data_train      = data.loc[data.index < test_start]
        data_test       = data

        return data_train, target_train, data_test, target_test

def quantile_negbin(qs, mean, dispersion=0):
    """ For `n` values in `qs` and `m` values in `mean`, computes the m by n matrix of `qs` quantiles for distributions with means `mean`.
    If dispersion is set to 0, a Poisson distribution is assumed, otherwise a Negative Binomial with corresponding dispersion parameter is used."""
    qs = np.array(qs).ravel()
    mean = np.array(mean).ravel()
    res = np.empty((len(mean), len(qs)))
    for i,mu in enumerate(mean):
        if dispersion == 0:
            # plot Poisson quantiles
            dist = scipy.stats.poisson(mu)
        else:
            # k = y
            n = 1/dispersion
            p = 1/(1+dispersion*mu)
            dist = scipy.stats.nbinom(n,p)
        for j,q in enumerate(qs):
            res[i,j] = dist.ppf(q)
    return res


def deviance_negbin(y, μ, α, saturated="NegativeBinomial"):
    if saturated=="NegativeBinomial":
        logp_sat = tt.where(y==0, np.zeros_like(y,dtype=np.float64), pm.NegativeBinomial.dist(mu=y, alpha=α).logp(y))
    elif saturated=="Poisson":
        logp_sat = tt.where(y==0, np.zeros_like(y,dtype=np.float64), pm.Poisson.dist(mu=y).logp(y))
    else:
        raise NotImplementedError()
    logp_mod = pm.NegativeBinomial.dist(mu=μ, alpha=α).logp(y)
    return (2*(logp_sat - logp_mod)).eval()

def dss(y, μ, var):
    return np.log(var) + (y-μ)**2/var

def pit_negbin(y, μ, α):
    return scipy.stats.nbinom.cdf(y, α, μ/(α + μ))

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


def make_axes_stack(ax, num, xoff, yoff, fig=None, down=True, sharex=False, sharey=False):
    if isinstance(ax, plt.Axes):
        bbox = ax.get_position()
        if fig is None:
            fig = ax.figure
        ax.set_zorder(num if down else 0)
        axes = [ax]
    elif isinstance(ax, gridspec.SubplotSpec):
        if fig is None:
            fig = plt.gcf()
        bbox = ax.get_position(fig)
        axes = [fig.add_axes(bbox, zorder = num if down else 0)]
    else:
        if fig is None:
            fig = plt.gcf()
        bbox = transforms.Bbox.from_bounds(ax[0], ax[1], ax[2]-ax[0], ax[3]-ax[1])
        axes = [fig.add_axes(bbox, zorder=num if down else 0)]

    for i in range(1,num):
        bbox = bbox.translated(xoff, yoff)
        ax = fig.add_axes(bbox, zorder=num-i if down else i, sharex=axes[0] if sharex else None, sharey=axes[0] if sharey else None)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_visible(False)
        axes.append(ax)
    return axes



def set_file_permissions(filename, uid, gid, permissions=0o660):
    os.chmod(filename, permissions)
    os.chown(filename, uid, gid)


def pairplot(df, labels={}, diagonal_kind="kde", lower_kind="kde", upper_kind="empty", spec=gridspec.GridSpec(1,1)[0],
xlabelrotation=0, ylabelrotation=90, ylabels=True, xlabels=True, xtickrotation=60,
fig=plt.gcf(), lower_kwargs={}, diagonal_kwargs={}, upper_kwargs={}, rasterized=False, tick_args={}):
    N = len(df.columns)
    axes = np.empty((N,N),dtype=object)

    g = gridspec.GridSpecFromSubplotSpec(N,N, subplot_spec=spec)
    fake_axes = {}
    for y in range(N):
        fake_axes[(y,0)] = plt.Subplot(fig, g[y,0])
        fake_axes[(y,0)].set_visible(False)
    for x in range(1,N):
        fake_axes[(0,x)] = plt.Subplot(fig, g[0,x])
        fake_axes[(0,x)].set_visible(False)

    for y,v2 in enumerate(df.columns):
        for x,v1 in enumerate(df.columns):
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

            if kind == "empty":
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

            axes[y,x].set_rasterized(rasterized)

    positive = np.all(df.values >= 0)

    for y in range(N):
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
