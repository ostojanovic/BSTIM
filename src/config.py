import itertools as it

diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))
