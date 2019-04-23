# -*- coding: utf-8 -*-
import itertools as it, pickle as pkl, os
from collections import OrderedDict
from sampling_utils import *
from shared_utils import *

diseases    = ["campylobacter", "rotavirus", "borreliosis"]
num_samples = 100
combinations = list(it.product(range(num_samples), diseases))
GID = int(os.environ["SGE_TASK_ID"])
num_sample, disease = combinations[GID-1]

filename = "../data/ia_effect_samples/{}_{}.pkl".format(disease, num_sample)

print("Running task {} - disease: {} - sample: {}\nWill create file {}".format(GID, disease, num_sample, filename))

        
with open('../data/counties/counties.pkl',"rb") as f:
    counties = pkl.load(f)


prediction_region = "bavaria" if disease=="borreliosis" else "germany"
parameters = OrderedDict()

# Load data
data = load_data(disease, prediction_region, counties)

times=uniform_times_by_week(data.index)
locs=uniform_locations_by_county(counties)
temporal_bfs = lambda x: bspline_bfs(x, np.array([0,0,1,2,3,4,5])*7*24*3600.0, 2)
spatial_bfs = lambda x: [gaussian_bf(x,σ) for σ in [6.25, 12.5, 25.0, 50.0]]

samp = IAEffectSampler(data, times, locs, temporal_bfs, spatial_bfs, num_tps = 10, time_horizon=5)
res = samp(data.index, data.columns)
results = {"ia_effects": res, "predicted week": data.index, "predicted county": data.columns}
with open(filename, "wb") as file:
    pkl.dump(results, file)


set_file_permissions(filename, uid=46836, gid=10033)
