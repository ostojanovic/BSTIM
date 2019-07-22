# -*- coding: utf-8 -*-
import pymc3 as pm, theano, theano.tensor as tt, numpy as np, pandas as pd, isoweek, pickle as pkl, datetime, time
from collections import OrderedDict
from matplotlib import pyplot as pp
from geo_utils import jacobian_sq
theano.config.compute_test_value = 'off' # BUG: may throw an error for flat RVs


def uniform_times_by_week(weeks, n=500):
    res = OrderedDict()
    for week in weeks:
        time_min = datetime.datetime.combine(isoweek.Week(*week).monday(), datetime.time.min)
        time_max = datetime.datetime.combine(isoweek.Week(*week).sunday(), datetime.time.max)
        res[week] = np.random.rand(n) * (time_max-time_min) + time_min
    return res

def uniform_locations_by_county(counties, n=500):
    res = OrderedDict()
    for (county_id, county) in counties.items():
        tp = county["testpoints"]
        if n == len(tp):
            res[county_id] = tp
        else:
            idx = np.random.choice(tp.shape[0], n, replace=n>len(tp))
            res[county_id] = tp[idx]
    return res


def sample_time_and_space(data, times_by_week, locations_by_county):
    n_total = data.sum().sum()
    t_all = np.empty((n_total,), dtype=object)
    x_all = np.empty((n_total,2))
    
    i = 0
    for (county_id, series) in data.iteritems():
        for (week, n) in series.iteritems():
            # draw n random times
            times = times_by_week[week]
            idx = np.random.choice(len(times), n)
            t_all[i:i+n] = times[idx]
            
            # draw n random locations
            locs = locations_by_county[county_id]
            idx = np.random.choice(locs.shape[0], n)
            x_all[i:i+n,:] = locs[idx,:]
            
            i += n
    
    return t_all, x_all

def gaussian_bf(dx, σ):
    σ = np.float32(σ)
    res = tt.zeros_like(dx)
    idx = (abs(dx) < np.float32(5)*σ)#.nonzero()
    return tt.set_subtensor(res[idx], tt.exp(np.float32(-0.5/(σ**2))*(dx[idx])**2)/np.float32(np.sqrt(2*np.pi*σ**2)))

def gaussian_gram(σ):
    return np.array([[np.power(2*np.pi*(a**2 + b**2), -0.5) for b in σ] for a in σ])

def bspline_bfs(x, knots, P):
    knots = knots.astype(np.float32)
    idx = ((x>=knots[0])&(x<knots[-1]))#.nonzero()
    xx = x[idx]
    
    N = {}
    for p in range(P+1):
        for i in range(len(knots)-1-p):
            if p==0:
                N[(i,p)] = tt.where((knots[i]<=xx)*(xx<knots[i+1]), 1.0, 0.0)
            else:
                N[(i,p)] =  (xx-knots[i])/(knots[i+p]-knots[i])*N[(i,p-1)] + \
                            (knots[i+p+1]-xx)/(knots[i+p+1]-knots[i+1])*N[(i+1,p-1)]
                            
    highest_level = []
    for i in range(len(knots)-1-P):
        res = tt.zeros_like(x)
        highest_level.append(tt.set_subtensor(res[idx], N[(i,P)]))
    return highest_level

def jacobian_sq(latitude, R = 6365.902):
    """
        jacobian_sq(latitude)
        
    Computes the "square root" (Cholesky factor) of the Jacobian of the cartesian projection from polar coordinates (in degrees longitude, latitude) onto cartesian coordinates (in km east/west, north/south) at a given latitude (the projection's Jacobian is invariante wrt. longitude).
    """
    return R*(np.pi/180.0)*(abs(tt.cos(tt.deg2rad(latitude)))*np.array([[1.0, 0.0], [0.0, 0.0]])+np.array([[0.0, 0.0],[0.0, 1.0]]))


def build_ia_bfs(temporal_bfs, spatial_bfs):
    x1 = tt.fmatrix("x1")
    t1 = tt.fvector("t1")
    # M = tt.fmatrix("M")
    x2 = tt.fmatrix("x2")
    t2 = tt.fvector("t2")

    lat = x1[:,1].mean()
    M = jacobian_sq(lat)**2
    
    # (x1,t1) are the to-be-predicted points, (x2,t2) the historic cases
    
    # spatial distance btw. each points in x1 and x2 with gramian M
    dx = tt.sqrt((x1.dot(M)*x1).sum(axis=1).reshape((-1,1)) + (x2.dot(M)*x2).sum(axis=1).reshape((1,-1)) - 2*x1.dot(M).dot(x2.T))
    
    # temporal distance btw. each times in t1 and t2
    dt = t1.reshape((-1,1)) - t2.reshape((1,-1))
    
    ft = tt.stack(temporal_bfs(dt.reshape((-1,))),axis=0)
    fx = tt.stack(spatial_bfs(dx.reshape((-1,))),axis=0)
    
    # aggregate contributions of all cases
    contrib = ft.dot(fx.T).reshape((-1,))/tt.cast(x1.shape[0], "float32")
    

                    
    return theano.function([t1,x1,t2,x2], contrib, allow_input_downcast=True)

class IAEffectSampler(object):
    def __init__(self, data, times_by_week, locations_by_county, temporal_bfs, spatial_bfs, num_tps = 10, time_horizon=5, verbose=True):
        self.ia_bfs = build_ia_bfs(temporal_bfs, spatial_bfs)
        self.times_by_week = times_by_week
        self.locations_by_county = locations_by_county
        self._to_timestamp = np.frompyfunc(datetime.datetime.timestamp, 1, 1)
        self.data = data
        self.num_tps = num_tps
        self.time_horizon = time_horizon
        self.num_features = len(temporal_bfs(tt.fmatrix("tmp")))*len(spatial_bfs(tt.fmatrix("tmp")))
        self.verbose = verbose
        
    def __call__(self, weeks, counties):
        res = np.zeros((len(weeks), len(counties), self.num_features), dtype=np.float32)
        for i,week in enumerate(weeks):
            for j,county in enumerate(counties):
                idx = ((isoweek.Week(*week)-self.time_horizon) <= self.data.index)*(self.data.index < week)
                # print("sampling week {} for county {} using data in range {}".format(week, county, idx))
                t_data, x_data = sample_time_and_space(self.data.iloc[idx], self.times_by_week, self.locations_by_county)
                t_pred, x_pred = sample_time_and_space(pd.DataFrame(self.num_tps, index=[week], columns=[county]), self.times_by_week, self.locations_by_county)
                res[i,j,:] = self.ia_bfs(self._to_timestamp(t_pred), x_pred, self._to_timestamp(t_data), x_data)
            frac = (i+1)/len(weeks)
            if self.verbose:
                print("⎹" + "█"*int(np.floor(frac*100)) + " ░▒▓█"[int(((frac*100)%1)*5)] + " "*int(np.ceil((1-frac)*100)) + "⎸ ({:.3}%)".format(100*frac), end="\r", flush=True)
        return res
