import theano
import re, pandas as pd, datetime, numpy as np, scipy as sp, pymc3 as pm, patsy as pt, theano.tensor as tt
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
                print("Warning: File {} not found!".format(filename))
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

    def __init__(self, trange, counties, ia_effect_filenames, num_ia=16, model=None, include_ia=True, include_eastwest=True, include_demographics=True, include_temporal=True, orthogonalize=False):
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
        
        self.Q = np.eye(16, dtype=np.float32)
        if orthogonalize:
            # transformation to orthogonalize IA features
            T = np.linalg.inv(np.linalg.cholesky(gaussian_gram([6.25,12.5,25.0,50.0]))).T
            for i in range(4):
                self.Q[i*4:(i+1)*4, i*4:(i+1)*4] = T


    def evaluate_features(self, weeks, counties):
        all_features = {}
        for group_name,features in self.features.items():
            group_features = {}
            for feature_name,feature in features.items():
                feature_matrix = feature(weeks, counties)
                group_features[feature_name] = pd.DataFrame(feature_matrix[:,:], index=weeks, columns=counties).stack()
            all_features[group_name] = pd.DataFrame([], index=pd.MultiIndex.from_product([weeks,counties]), columns=[]) if len(group_features)==0 else pd.DataFrame(group_features)
        return all_features

    def init_model(self, target):
        weeks,counties = target.index, target.columns

        # extract features
        features = self.evaluate_features(weeks, counties)
        Y_obs = target.stack().values.astype(np.float32)
        T_S = features["temporal_seasonal"].values.astype(np.float32)
        T_T = features["temporal_trend"].values.astype(np.float32)
        TS = features["spatiotemporal"].values.astype(np.float32)
        S = features["spatial"].values.astype(np.float32)

        log_exposure = np.log(features["exposure"].values.astype(np.float32).ravel())

        # extract dimensions
        num_obs = np.prod(target.shape)
        num_t_s = T_S.shape[1]
        num_t_t = T_T.shape[1]
        num_ts = TS.shape[1]
        num_s = S.shape[1]


        with pm.Model() as self.model:
            # interaction effects are generated externally -> flat prior
            IA    = pm.Flat("IA", testval=np.ones((num_obs, self.num_ia)),shape=(num_obs, self.num_ia))

            # priors
            #δ = 1/√α
            δ     = pm.HalfCauchy("δ", 10, testval=1.0)
            α     = pm.Deterministic("α", np.float32(1.0)/δ)
            W_ia  = pm.Normal("W_ia", mu=0, sd=10, testval=np.zeros(self.num_ia), shape=self.num_ia)
            W_t_s = pm.Normal("W_t_s", mu=0, sd=10, testval=np.zeros(num_t_s), shape=num_t_s)
            W_t_t = pm.Normal("W_t_t", mu=0, sd=10, testval=np.zeros(num_t_t), shape=num_t_t)
            W_ts  = pm.Normal("W_ts", mu=0, sd=10, testval=np.zeros(num_ts), shape=num_ts)
            W_s   = pm.Normal("W_s", mu=0, sd=10, testval=np.zeros(num_s), shape=num_s)
            self.param_names = ["δ", "W_ia", "W_t_s", "W_t_t", "W_ts", "W_s"]
            self.params = [δ, W_ia, W_t_s, W_t_t, W_ts, W_s]

            # calculate interaction effect
            IA_ef = tt.dot(tt.dot(IA, self.Q), W_ia)

            # calculate mean rates
            μ = pm.Deterministic("μ", 
                # (1.0+tt.exp(IA_ef))*
                tt.exp(IA_ef + tt.dot(T_S, W_t_s) + tt.dot(T_T, W_t_t) + tt.dot(TS, W_ts) + tt.dot(S, W_s) + log_exposure)
            )

            # constrain to observations
            pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)


    def sample_parameters(self, target, n_init=100, samples=1000, chains=None, cores=8, init="advi", target_accept=0.8, max_treedepth=10, **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """
        # model = self.model(target)

        self.init_model(target)

        if chains is None:
            chains = max(2,cores)
            

        with self.model:
            # run!
            ia_effect_loader = IAEffectLoader(self.model.IA, self.ia_effect_filenames, target.index, target.columns)
            nuts = pm.step_methods.NUTS(vars=self.params, target_accept=target_accept, max_treedepth=max_treedepth)
            steps = (([ia_effect_loader] if self.include_ia else [] ) + [nuts] )
            trace = pm.sample(samples, steps, chains=chains, cores=cores, compute_convergence_checks=False, **kwargs)
            # trace = pm.sample(0, steps, tune=samples+tune, discard_tuned_samples=False, chains=chains, cores=cores, compute_convergence_checks=False, **kwargs)
            # trace = trace[tune:]
        return trace

    def sample_predictions(self, target_weeks, target_counties, parameters, init="auto"):
        # extract features
        features = self.evaluate_features(target_weeks, target_counties)

        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        TS = features["spatiotemporal"].values
        S = features["spatial"].values
        log_exposure = np.log(features["exposure"].values.ravel())

        # extract coefficient samples
        α = parameters["α"]
        W_ia = parameters["W_ia"]
        W_t_s = parameters["W_t_s"]
        W_t_t = parameters["W_t_t"]
        W_ts = parameters["W_ts"]
        W_s = parameters["W_s"]
        

        ia_l = IAEffectLoader(None, self.ia_effect_filenames, target_weeks, target_counties)

        num_predictions = len(target_weeks)*len(target_counties)
        num_parameter_samples = α.size
        y = np.zeros((num_parameter_samples, num_predictions), dtype=int)
        μ = np.zeros((num_parameter_samples, num_predictions), dtype=np.float32)

        for i in range(num_parameter_samples):
            IA_ef = np.dot(np.dot(ia_l.samples[np.random.choice(len(ia_l.samples))], self.Q), W_ia[i])
            # μ[i,:] = (1.0+np.exp(IA_ef))*np.exp(np.dot(T_S, W_t_s[i]) + np.dot(T_T, W_t_t[i]) + np.dot(TS, W_ts[i]) + np.dot(S, W_s[i]) + log_exposure)
            μ[i,:] = np.exp(IA_ef + np.dot(T_S, W_t_s[i]) + np.dot(T_T, W_t_t[i]) + np.dot(TS, W_ts[i]) + np.dot(S, W_s[i]) + log_exposure)
            y[i,:] = pm.NegativeBinomial.dist(mu=μ[i,:], alpha=α[i]).random()

        return {"y": y, "μ": μ, "α": α}
