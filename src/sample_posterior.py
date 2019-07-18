# from ModelInstance import ModelInstance
from shared_utils import load_data, split_data
from collections import OrderedDict
import pickle as pkl, time, numpy as np, scipy as sp, pandas as pd
import pymc3 as pm
import itertools as it
import os
import isoweek
from BaseModel import BaseModel

diseases    = ["campylobacter", "rotavirus", "borreliosis"]
combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))

i = int(os.environ["SGE_TASK_ID"])-1

num_samples = 500
num_chains = 4
num_cores = num_chains

model_complexity, disease = combinations[i]
use_age, use_eastwest     = combinations_age_eastwest[model_complexity]
prediction_region         = "bavaria" if disease=="borreliosis" else "germany"

filename_params = "../data/mcmc_samples/parameters_{}_{}_{}".format(disease, use_age, use_eastwest)
filename_pred = "../data/mcmc_samples/predictions_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)
filename_model = "../data/mcmc_samples/model_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)

# Load data
with open('../data/counties/counties.pkl',"rb") as f:
    county_info = pkl.load(f)
data = load_data(disease, prediction_region, county_info)
data_train, target_train, data_test, target_test = split_data(data)

tspan = (target_train.index[0],target_train.index[-1])

print("training for {} in {} with model complexity {} from {} to {}\nWill create files {}, {} and {}".format(disease, prediction_region, model_complexity,*tspan, filename_params, filename_pred, filename_model))

model = BaseModel(tspan, county_info, ["../data/ia_effect_samples/{}_{}.pkl".format(disease, i) for i in range(100)], include_eastwest=use_eastwest, include_demographics=use_age)

print("Sampling parameters on the training set.")
trace = model.sample_parameters(target_train, samples=num_samples, tune=100, target_accept=0.95, max_treedepth=15, chains=num_chains, cores=num_cores)
# with open(filename_params, 'wb') as f:
#    pkl.dump(trace, f)

with open(filename_model, "wb") as f:
   pkl.dump(model.model, f)

with model.model:
   pm.save_trace(trace, filename_params)

print("Sampling predictions on the testing set.")
pred = model.sample_predictions(target_test.index, target_test.columns, trace)
with open(filename_pred, 'wb') as f:
   pkl.dump(pred, f)

# for file in [filename_params, filename_pred]:
#     set_file_permissions(file, uid=46836, gid=10033)
