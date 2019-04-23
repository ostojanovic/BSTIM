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

"""
Evaluating model for campylobacter...
Results: 
     WAIC   pWAIC  dWAIC weight      SE   dSE var_warn
B  420139  122.76      0      1   629.9     0        1
A  420140  122.92   0.36      0  629.95   1.1        1
C  420151  127.04  11.97      0  630.13  1.23        1

Evaluating model for rotavirus...
Results: 
     WAIC   pWAIC dWAIC weight      SE   dSE var_warn
C  338735  142.38     0      1  944.44     0        1
A  338736  143.84  1.04      0  944.36  1.33        1
B  338736  143.79  1.39      0   944.3  1.33        1

Evaluating model for borreliosis...
Results: 
      WAIC  pWAIC dWAIC weight      SE   dSE var_warn
A  30807.4   28.2     0      1  316.75     0        0
B  30807.8  28.35   0.4      0  316.75  0.63        0
C    30810  28.88  2.56      0  316.82  0.57        0
"""
