from config import *
from shared_utils import *
import pickle as pkl
from collections import OrderedDict

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
    if disease=="borreliosis":
        prediction_region = "bavaria"
        use_eastwest = False
    else:
        prediction_region = "germany"

    res = load_pred(disease, use_age, use_eastwest)

    with open('../data/hhh4_results_{}.pkl'.format(disease),"rb") as f:
        res_hhh4 = pkl.load(f)

    data = load_data(disease, prediction_region, counties)
    data = data[data.index < parse_yearweek("2018-KW1")]
    if disease == "borreliosis":
        data = data[data.index >= parse_yearweek("2013-KW1")]
    _, _, _, target = split_data(data)
    county_ids = target.columns

    summary = {}
    # hhh4

    for name in ["our model", "hhh4 model"]:
        summary[name] = {}
        for measure,f in measures.items():
            print("Evaluating {} for disease {}, measure {}".format(name, disease, measure))
            if name == "our model":
                measure_df = pd.DataFrame(f(target.values.astype(np.float32).reshape((1,-1)).repeat(res["y"].shape[0], axis=0), res["μ"].astype(np.float32), res["α"].astype(np.float32).reshape((-1,1))).mean(axis=0).reshape(target.shape), index=target.index, columns=target.columns)
            else:
                measure_df = pd.DataFrame(f(target.values.astype(np.float32).ravel(), res_hhh4["test prediction mean"].values.astype(np.float32).ravel(), np.float32(1.0/res_hhh4["test alpha"])).reshape(target.shape), index=target.index, columns=target.columns)
        
            summary[name][measure] = measure_df
            summary[name][measure + " mean"] = np.mean(measure_df.mean())
            summary[name][measure + " sd"] = np.std(measure_df.mean())

    with open("../data/measures_{}_summary.pkl".format(disease),"wb") as f:
        pkl.dump(summary, f)
    
    del summary
