import pickle as pkl
import numpy as np
from shared_utils import dss, deviance_negbin, load_data, split_data, parse_yearweek
from collections import OrderedDict
import pandas as pd

diseases = ["campylobacter", "rotavirus", "borreliosis"]
measures = {
    "deviance": (lambda target_val, pred_val, alpha_val: deviance_negbin(target_val, pred_val, alpha_val)),
    "DS score": (lambda target_val, pred_val, alpha_val: dss(target_val, pred_val, pred_val+pred_val**2/alpha_val))
}

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)


with open('../data/counties/counties.pkl',"rb") as f:
    counties = pkl.load(f)

summary = OrderedDict()

for i,disease in enumerate(diseases):
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
    filename_pred = "../data/mcmc_samples/predictions_{}_{}_{}.pkl".format(disease, use_age, use_eastwest)
    prediction_region = "bavaria" if disease=="borreliosis" else "germany"

    with open(filename_pred,"rb") as f:
        res = pkl.load(f)

    mean_predicted_μ = np.reshape(res['μ'],(800,104,-1)).mean(axis=0)
    mean_predicted_α = res['α'].mean()

    with open('../data/hhh4_results_{}.pkl'.format(disease),"rb") as f:
        res_hhh4 = pkl.load(f)

    data = load_data(disease, prediction_region, counties)
    data = data[data.index < parse_yearweek("2018-KW1")]
    if disease == "borreliosis":
        data = data[data.index >= parse_yearweek("2013-KW1")]
    _, _, _, target = split_data(data)
    county_ids = target.columns

    models = {
        "our model": (pd.DataFrame(mean_predicted_μ, columns=target.columns, index=target.index), pd.Series(np.repeat(mean_predicted_α, target.shape[0]), index=target.index)),
        "hhh4 model": (res_hhh4["test prediction mean"], pd.Series(np.repeat(1.0/res_hhh4["test alpha"], target.shape[0]), index=target.index)),
    }

    assert np.all(models["our model"][0].columns == models["hhh4 model"][0].columns), "Column names don't match!"

    summary[disease] = {}
    for model,(prediction, alpha) in models.items():
        summary[disease][model] = {}

        for measure,f in measures.items():
            measure_df = pd.DataFrame(f(target.values.ravel(), prediction.values.ravel(), alpha.values.repeat(target.shape[1])).reshape(target.shape), index=target.index, columns=target.columns)
            summary[disease][model][measure] = measure_df
            summary[disease][model][measure + " mean"] = np.mean(measure_df.mean())
            summary[disease][model][measure + " sd"] = np.std(measure_df.mean())

with open('../data/measures_summary.pkl',"wb") as f:
    pkl.dump(summary, f)
