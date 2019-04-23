import re, pandas as pd, datetime, numpy as np, scipy as sp, pymc3 as pm, patsy as pt, theano, theano.tensor as tt
theano.config.compute_test_value = 'off' # BUG: may throw an error for flat RVs
from collections import OrderedDict
from sampling_utils import *


class SpatioTemporalFeature(object):
    def __init__(self):
        self._call_ = np.frompyfunc(self.call, 2, 1)

    def __call__(self, times, locations):
        return self._call_(np.asarray(times).reshape((-1,1)), np.asarray(locations).reshape((1,-1))).astype(np.float32)

class SpatioTemporalYearlyDemographicsFeature(SpatioTemporalFeature):
    def __init__(self, county_dict, group, scale=1.0):
        self.dict = {
            (year, county): val*scale
            for county,values in county_dict.items()
            for (g, year),val in values["demographics"].items()
            if g == group
        }
        super().__init__()

    def call(self, yearweek,county):
        return self.dict.get((yearweek[0],county))

class SpatialEastWestFeature(SpatioTemporalFeature):
    def __init__(self, county_dict):
        self.dict = {
            county: 1.0 if "east" in values["region"] else (0.5 if "berlin" in values["region"] else 0.0)
            for county,values in county_dict.items()
        }
        super().__init__()

    def call(self, yearweek,county):
        return self.dict.get(county)

class TemporalFourierFeature(SpatioTemporalFeature):
    def __init__(self, i, t0, scale):
        self.t0 = t0
        self.scale = scale
        self.τ = (i//2 + 1)*2*np.pi
        self.fun = np.sin if (i%2)==0 else np.cos
        super().__init__()

    def call(self, t, x):
        return self.fun((t-self.t0)/self.scale*self.τ)

class TemporalSigmoidFeature(SpatioTemporalFeature):
    def __init__(self, t0, scale):
        self.t0 = t0
        self.scale = scale
        super().__init__()

    def call(self, t, x):
        return sp.special.expit((t-self.t0)/self.scale)


class IAEffectLoader(object):
    generates_stats = False
    def __init__(self, var, filenames, weeks, counties):
        self.vars = [var]
        self.samples = []
        for filename in filenames:
            try:
                with open(filename, "rb") as f:
                    tmp=pkl.load(f)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(e)
            else:
                m = tmp["ia_effects"]
                ws = list(tmp["predicted week"])
                cs = list(tmp["predicted county"])
                w_idx = np.array([ws.index(w) for w in weeks]).reshape((-1,1))
                c_idx = np.array([cs.index(c) for c in counties])
                self.samples.append(np.moveaxis(m[w_idx,c_idx,:], -1, 0).reshape((m.shape[-1], -1)).T)

    def step(self, point):
        new = point.copy()
        # res = new[self.vars[0].name]
        new_res = self.samples[np.random.choice(len(self.samples))]
        new[self.vars[0].name] = new_res
        return new

    def stop_tuning(*args):
        pass

    @property
    def vars_shape_dtype(self):
        shape_dtypes = {}
        for var in self.vars:
            dtype = np.dtype(var.dtype)
            shape = var.dshape
            shape_dtypes[var.name] = (shape, dtype)
        return shape_dtypes

class BaseModel(object):
    """
    Model for disease prediction.

    The model has 4 types of features (predictor variables):
    * temporal (functions of time)
    * spatial (functions of space, i.e. longitude, latitude)
    * county_specific (functions of time and space, i.e. longitude, latitude)
    * interaction effects (functions of distance in time and space relative to each datapoint)
    """

    def __init__(self, trange, counties, ia_effect_filenames, num_ia=16, model=None, include_ia=True, include_eastwest=True, include_demographics=True, include_temporal=True):
        self.county_info = counties
        self.ia_effect_filenames = ia_effect_filenames
        self.num_ia = num_ia if include_ia else 0
        self.include_ia = include_ia
        self.include_eastwest = include_eastwest
        self.include_demographics = include_demographics
        self.include_temporal = include_temporal
        self.trange = trange
        
        first_year=self.trange[0][0]
        last_year=self.trange[1][0]
        self.features = {
                "temporal_seasonal": {"temporal_fourier_{}".format(i): TemporalFourierFeature(i, isoweek.Week(first_year,1), 52.1775) for i in range(4)} if self.include_temporal else {},
                "temporal_trend": {"temporal_sigmoid_{}".format(i): TemporalSigmoidFeature(isoweek.Week(i,1), 2.0) for i in range(first_year,last_year+1)} if self.include_temporal else {},
                "spatiotemporal": {"demographic_{}".format(group): SpatioTemporalYearlyDemographicsFeature(self.county_info, group) for group in ["[0-5)", "[5-20)", "[20-65)"]} if self.include_demographics else {},
                "spatial": {"eastwest": SpatialEastWestFeature(self.county_info)} if self.include_eastwest else {},
                "exposure": {"exposure": SpatioTemporalYearlyDemographicsFeature(self.county_info, "total", 1.0/100000)}
            }
    
    def evaluate_features(self, weeks, counties):
        all_features = {}
        for group_name,features in self.features.items():
            group_features = {}
            for feature_name,feature in features.items():
                feature_matrix = feature(weeks, counties)
                group_features[feature_name] = pd.DataFrame(feature_matrix[:,:], index=weeks, columns=counties).stack()
            all_features[group_name] = pd.DataFrame(group_features)
        return all_features

    def model(self, target):
        weeks,counties = target.index, target.columns

        features = self.evaluate_features(weeks, counties)
                
        Y = target.stack().values.astype(np.float32)
        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        TS = features["spatiotemporal"].values
        S = features["spatial"].values
        exposure = features["exposure"].values.ravel()
            
        with pm.Model() as model:
            α     = pm.HalfCauchy("α", beta=2.0)
            IA    = pm.Flat("IA", shape=(len(Y),self.num_ia))

            W_ia  = pm.HalfNormal("W_ia", sd=1, shape=self.num_ia)
            W_t_s = pm.Normal("W_t_s", mu=0, sd=1, shape=T_S.shape[1])
            W_t_t = pm.Normal("W_t_t", mu=0, sd=1, shape=T_T.shape[1])

            s = tt.dot(IA, W_ia) + tt.dot(T_S, W_t_s) + tt.dot(T_T, W_t_t)

            if TS.shape[1]!=0:
                W_ts  = pm.Normal("W_ts", mu=0, sd=1, shape=TS.shape[1])
                s += tt.dot(TS, W_ts)
                
            if S.shape[1]!=0:
                W_s   = pm.Normal("W_s", mu=0, sd=1, shape=S.shape[1])
                s += tt.dot(S, W_s)

            μ = pm.Deterministic("μ", tt.exp(s)*exposure)
            pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y)
        
        return model

    def sample_parameters(self, target, samples=1000, chains=None, cores=8, init="auto", **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """
        model = self.model(target)
        
        if chains is None:
            chains = max(2,cores)
        
        with model:
            ia_effect_loader = IAEffectLoader(model.IA, self.ia_effect_filenames, target.index, target.columns)
            
            vars = [model.α, model.W_ia, model.W_t_s, model.W_t_t]
            if hasattr(model,"W_ts"):
                vars += [model.W_ts]
            if hasattr(model,"W_s"):
                vars += [model.W_s]

            steps = ([ia_effect_loader] if self.include_ia else [] ) + \
                    [pm.step_methods.NUTS(vars=vars)]
            trace = pm.sample(samples, steps, chains=chains, cores=cores, init=init, compute_convergence_checks=False, **kwargs)

        return trace

    def sample_predictions(self, target_weeks, target_counties, parameters, init="auto"):
        features = self.evaluate_features(target_weeks, target_counties)
                
        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        TS = features["spatiotemporal"].values
        S = features["spatial"].values
        exposure = features["exposure"].values.reshape((-1,1))

        α = parameters["α"].reshape((1,-1))
        W_ia = parameters["W_ia"]
        W_t_s = parameters["W_t_s"]
        W_t_t = parameters["W_t_t"]
        W_ts = parameters["W_ts"]
        W_s = parameters["W_s"]
        
        num_predictions = len(target_weeks)*len(target_counties)
        num_parameter_samples = α.size
        with pm.Model() as model:
            IA = pm.Flat("IA", shape=(num_predictions,self.num_ia))
            ia_effect_loader = IAEffectLoader(model.IA, self.ia_effect_filenames, target_weeks, target_counties)
            IA_trace = pm.sample(num_parameter_samples, ia_effect_loader, chains=1, cores=1, init=init, compute_convergence_checks=False)["IA"]
        
        μs = np.exp(np.dot(T_S, W_t_s.T) + np.dot(T_T, W_t_t.T) + np.dot(TS, W_ts.T) + np.dot(S, W_s.T) + (IA_trace*W_ia[:,np.newaxis,:]).sum(axis=-1).T)*exposure
        ys = pm.NegativeBinomial.dist(mu=μs, alpha=α).random()
        
        new_trace = {}
        for varname in parameters.varnames:
            new_trace[varname] = parameters[varname]
        new_trace["IA"] = IA_trace.T
        new_trace["μ"] = μs.T
        new_trace["Y"] = ys.T
        
        return new_trace
