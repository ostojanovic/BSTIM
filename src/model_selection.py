from shared_utils import load_data, split_data
import pickle as pkl
import pymc3 as pm
from BaseModel import BaseModel

diseases    = ["campylobacter", "rotavirus", "borreliosis"]
combinations_age_eastwest = [(False,False),(False,True),(True,True)]
age_eastwest_by_name = dict(zip(["A","B","C"],combinations_age_eastwest))

with open('../data/counties/counties.pkl',"rb") as f:
    county_info = pkl.load(f)

results = {}
best_model = {}
for disease in diseases:
    print("Evaluating model for {}...".format(disease))
    prediction_region = "bavaria" if disease=="borreliosis" else "germany"
    data = load_data(disease, prediction_region, county_info)
    data_train, target_train, data_test, target_test = split_data(data)
    tspan = (target_train.index[0],target_train.index[-1])
    models = {}
    for (name,(use_age,use_eastwest)) in age_eastwest_by_name.items():
        # load sample trace
        filename_pred = "../data/mcmc_samples/parameters_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)
        with open(filename_pred, 'rb') as f:
           trace = pkl.load(f)
        
        # construct model
        model = BaseModel(tspan, county_info, ["../data/ia_effect_samples/{}_{}.pkl".format(disease, i) for i in range(100)], include_eastwest=use_eastwest, include_demographics=use_age).model(target_train)
        
        model.name = name
        models[model] = trace
    # do model selection
    results[disease] = pm.compare(models,ic="WAIC")
    print("Results: ")
    print(results[disease])
    
    name = results[disease].iloc[0].name
    use_age, use_eastwest = age_eastwest_by_name[name]
    best_model[disease] = {"name": name, "use_age": use_age, "use_eastwest": use_eastwest, "comparison": results[disease]}

with open('../data/comparison.pkl',"wb") as f:
    pkl.dump(best_model, f)
