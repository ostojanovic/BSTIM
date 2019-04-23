from ModelInstance import ModelInstance
from shared_utils import load_data, split_data, parse_yearweek, quantile_negbin
import pickle as pkl, os, pandas as pd, numpy as np
import pymc3 as pm

diseases    = ["campylobacter", "rotavirus", "borreliosis"]

# Load data
with open('../data/counties/counties.pkl',"rb") as f:
    counties = pkl.load(f)
county_interactions = np.load("../data/counties/interaction_effects.npy")

# Test
# We retrain and test week by week
for disease in diseases:
    print("testing for {}".format(disease))
    prediction_region = "bavaria" if disease=="borreliosis" else "germany"

    data = load_data(disease, prediction_region, counties)
    data = data[data.index < parse_yearweek("2018-KW1")]
    if disease == "borreliosis":
        data = data[data.index >= parse_yearweek("2013-KW1")]
        
    data_train, target_train, data_test, target_test = split_data(data)

    all_data = pd.concat([data_train, data_test])
    all_target = pd.concat([target_train, target_test])


    with open("../data/results_{}.pkl".format(disease), "rb") as file:
        results = pkl.load(file)
    
    model_complexity, model, parameters = results["best_model_complexity"], results["best_model"], results["best_parameters"]
    use_age,use_eastwest = [(False,False),(False,True),(True,True)][model_complexity]
    # parameters = parameters.mean()
    parameters=None

    # ds = best_model[disease]["model"]
    mean_prediction_test = np.zeros((target_test.shape[0], target_test.shape[1]), dtype=np.float)
    alpha_test = np.zeros(target_test.shape[0])
    perc_25_prediction_test = np.zeros_like(mean_prediction_test)
    perc_75_prediction_test = np.zeros_like(mean_prediction_test)
    for i in range(len(target_test)):
        #retrain model on all the available data before the week we want to predict using the optimal parameters
        all_available_data = all_data.iloc[all_data.index<target_test.index[i]]
        all_available_target = all_target.iloc[all_target.index<target_test.index[i]]

        ds = ModelInstance(
          tspan=(all_available_target.index[0],all_available_target.index[-1]),
          region=prediction_region,
          counties=counties,
          county_interactions=county_interactions,
          use_age=use_age,
          use_eastwest=use_eastwest,
          # model=model,
        )
        
        # use previously computed parameters as a starting point for gradient 
        # descent and add zeros for the potential new trend parameters
        if parameters is not None:
            if len(parameters["W_t"]) < ds.num_t:
                parameters["W_t"] = np.hstack((parameters["W_t"],np.zeros(ds.num_t-len(parameters["W_t"]))))
        
        print("Retrain on data from {} to {} (targets from {} to {})...".format(all_available_data.index[0], all_available_data.index[-1], all_available_target.index[0], all_available_target.index[-1]))
        parameters = ds.find_MAP(target_values=all_available_target, data=all_available_data, start=parameters)

        print("\tPredict for {}...".format(target_test.index[i]))
        # samples = ds.sample_predictions(parameters, target_test.index[i:i+1], target_test.columns, all_available_data, samples=1000)
        samples = ds.sample_mean([parameters], target_test.index[i:i+1], target_test.columns, all_available_data, samples=1)

        s = samples.shape
        alpha_test[i] = parameters["α"]
        mean_prediction_test[i,:] = np.mean(samples.reshape(s[0], -1), axis=0).reshape(s[1:])
        qs = quantile_negbin([0.25, 0.75], mean_prediction_test[i,:], dispersion=1.0/parameters["α"])
        # perc_25, perc_75 = map(lambda x: x.reshape(s[1:]), np.percentile(samples.reshape(s[0], -1), [25,75], axis=0))
        perc_25_prediction_test[i,:] = qs[:,0]
        perc_75_prediction_test[i,:] = qs[:,1]
    results["test alpha"]          = pd.Series(alpha_test, index=target_test.index)
    results["test prediction mean"]= pd.DataFrame(mean_prediction_test, columns=target_test.columns, index=target_test.index)
    results["test prediction 25%"] = pd.DataFrame(perc_25_prediction_test, columns=target_test.columns, index=target_test.index)
    results["test prediction 75%"] = pd.DataFrame(perc_75_prediction_test, columns=target_test.columns, index=target_test.index)
    results["test target"] = target_test
    results["test model"] = ds
    results["test parameters"] = [parameters]
    
    
    with open("../data/test_results_{}.pkl".format(disease), "wb") as file:
        pkl.dump(results, file)
