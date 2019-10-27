import itertools as it

diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))

prior_scales = [0.625, 2.5, 10.0, 40.0, 160.0]
sensitivity_analysis_combinations = list(it.product(prior_scales, [2], diseases))
