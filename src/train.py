from ModelInstance import ModelInstance
from shared_utils import load_data, split_data
from collections import OrderedDict
import pickle as pkl, time, numpy as np, pandas as pd
import pymc3 as pm
import itertools as it

diseases    = ["campylobacter", "rotavirus", "borreliosis"]
combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))

for disease in diseases:
    prediction_region = "bavaria" if disease=="borreliosis" else "germany"
    parameters = OrderedDict()
    
    for model_complexity in range(3):
        use_age, use_eastwest     = combinations_age_eastwest[model_complexity]

        # Load data
        with open('../data/counties/counties.pkl',"rb") as f:
            counties = pkl.load(f)
        county_interactions = np.load("../data/counties/interaction_effects.npy")

        data = load_data(disease, prediction_region, counties)
        data_train, target_train, _, _ = split_data(data)

        tspan = (target_train.index[0],target_train.index[-1])
        print("training for {} in {} with model complexity {} from {} to {}".format(disease, prediction_region, model_complexity,*tspan))

        ds = ModelInstance(
            tspan=tspan,
            region=prediction_region,
            counties=counties,
            county_interactions=county_interactions,
            use_age=use_age,
            use_eastwest=use_eastwest
        )

        tic = time.time()
        parameters[ds.model] = ds.sample_parameters(target_values=target_train, data=data_train, use_MAP=False, samples=500, tune=500, cores=7)
        ds.model.name = "ABC"[model_complexity]
        toc = time.time()
        print("Took {:.2f} seconds for MAP estimation.".format(toc-tic))
    comp = pm.compare(parameters, ic="WAIC")
    comp.index.name = "Model Complexity"
    
    best_idx = "ABC".index(comp.index[0])
    best_model, best_parameters = list(parameters.items())[best_idx]
    results = {"comparison": comp, "best_model_complexity": best_idx, "best_model": best_model, "best_parameters": best_parameters}

    with open("../data/results_{}.pkl".format(disease), "wb") as file:
        pkl.dump(results, file)


# with open("../data/results.pkl".format(disease, model_complexity), "wb") as file:
#     pkl.dump({"parameters": parameters, "model": ds.model}, file)
