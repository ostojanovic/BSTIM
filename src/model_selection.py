from config import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import datetime, pickle as pkl, numpy as np, matplotlib, pandas as pd, pymc3 as pm
from pymc3.stats import quantiles
from shared_utils import *
from sampling_utils import *
from matplotlib import rc
from collections import OrderedDict
import isoweek
import gc
import matplotlib.patheffects as PathEffects
from matplotlib.gridspec import SubplotSpec, GridSpec, GridSpecFromSubplotSpec
from BaseModel import BaseModel
import os

age_eastwest_by_name = dict(zip(["A","B","C"],combinations_age_eastwest))

with open('../data/counties/counties.pkl',"rb") as f:
    county_info = pkl.load(f)

results = {}
best_model = {}
for disease in diseases:
    print("Evaluating model for {}...".format(disease))
    if disease=="borreliosis":
       prediction_region = "bavaria"
       use_eastwest = False
    else:
       prediction_region = "germany"
       
    data = load_data(disease, prediction_region, county_info)
    data_train, target_train, data_test, target_test = split_data(data)
    tspan = (target_train.index[0],target_train.index[-1])
    models = {}
    for (name,(use_age,use_eastwest)) in age_eastwest_by_name.items():
        # load sample trace
        trace = load_trace(disease, use_age, use_eastwest)
        
        # load model
        model = load_model(disease, use_age, use_eastwest)
        
        model.name = name
        models[model] = trace
    # do model selection
    results[disease] = pm.compare(models,ic="WAIC")
    print("Results: ")
    print(results[disease])
    del models
    
    name = results[disease].iloc[0].name
    use_age, use_eastwest = age_eastwest_by_name[name]
    best_model[disease] = {"name": name, "use_age": use_age, "use_eastwest": use_eastwest, "comparison": results[disease]}

with open('../data/comparison.pkl',"wb") as f:
    pkl.dump(best_model, f)
