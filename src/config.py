import matplotlib
import itertools as it
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams["font.family"] = "Bitstream Charter"

matplotlib.rcParams['xtick.labelsize'] = 16
matplotlib.rcParams['ytick.labelsize'] = 16
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['axes.titlesize'] = 22

diseases = ["campylobacter", "rotavirus", "borreliosis"]
prediction_regions = ["germany", "bavaria"]

combinations_age_eastwest = [(False,False),(False,True),(True,True)]
combinations = list(it.product(range(len(combinations_age_eastwest)), diseases))
