from matplotlib import pyplot as plt
from config import *
from shared_utils import *
from plot_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict

diseases = ["campylobacter", "rotavirus", "borreliosis"]

with open('../data/counties/counties.pkl',"rb") as f:
    counties = pkl.load(f)

xlim = (5.5,15.5)
ylim = (47,56)

countyByName = OrderedDict([('Düsseldorf', '05111'),('Recklinghausen', '05562'),
                            ("Hannover", "03241"), ("Hamburg", "02000"),
                            ("Berlin-Mitte", "11001"), ("Osnabrück", "03404"),
                            ("Frankfurt (Main)", "06412"),
                            ("Görlitz", "14626"), ("Stuttgart","08111"),
                            ("Potsdam", "12054"), ("Köln", "05315"),
                            ("Aachen", "05334"), ("Rostock", "13003"),
                            ("Flensburg", "01001"), ("Frankfurt (Oder)", "12053"),
                            ("Lübeck", "01003"),("Münster", "05515"),
                            ("Berlin Neukölln", "11008"), ('Göttingen', "03159"),
                            ("Cottbus", "12052"), ("Erlangen", "09562"),
                            ("Regensburg", "09362"), ("Bayreuth", "09472"),
                            ("Bautzen", "14625"), ('Nürnberg', '09564'),
                            ('München', '09162'), ("Würzburg", "09679"),
                            ("Deggendorf", "09271"), ("Ansbach", "09571"),
                            ("Rottal-Inn", "09277"), ("Passau", "09275"),
                            ("Schwabach", "09565"), ("Memmingen", "09764"),
                            ("Erlangen-Höchstadt", "09572"), ("Nürnberger Land", "09574"),
                            ('Roth', "09576"), ('Starnberg', "09188"),
                            ('Berchtesgadener Land', "09172"), ('Schweinfurt', "09678"),
                            ("Augsburg","09772" ), ('Neustadt a.d.Waldnaab', "09374"),
                            ("Fürstenfeldbruck", "09179"), ('Rosenheim', "09187"),
                            ("Straubing", "09263"), ("Erding", "09177"),
                            ("Tirschenreuth", "09377"), ('Miltenberg', "09676"),
                            ('Neumarkt i.d.OPf.', "09373")])

plot_county_names = {"campylobacter": ["Düsseldorf", "Recklinghausen", "Hannover", "München",
                                        "Hamburg", "Berlin-Mitte", "Osnabrück", "Frankfurt (Main)",
                                        "Görlitz", "Stuttgart", "Potsdam", "Köln", "Aachen", "Rostock",
                                        "Flensburg", "Frankfurt (Oder)", "Lübeck", "Münster", "Berlin Neukölln",
                                        "Göttingen", "Cottbus", "Erlangen", "Regensburg", "Bayreuth", "Nürnberg"],
                    "rotavirus": ["Bautzen", "Hannover", "München", "Hamburg", "Düsseldorf", "Recklinghausen",
                                   "Berlin-Mitte", "Frankfurt (Main)", "Görlitz", "Stuttgart", "Potsdam",
                                   "Köln", "Aachen", "Rostock", "Flensburg", "Frankfurt (Oder)", "Lübeck", "Münster",
                                   "Berlin Neukölln", "Göttingen", "Cottbus", "Erlangen", "Regensburg", "Bayreuth", "Nürnberg"],
                    "borreliosis": ["Erlangen", "Regensburg", "Bayreuth", "Würzburg", "Deggendorf",
                                    "Ansbach", "Rottal-Inn", "Passau", "Schwabach", "Memmingen", "Erlangen-Höchstadt", "Nürnberger Land",
                                    'Roth', 'Starnberg', 'Berchtesgadener Land', 'Schweinfurt', "Augsburg", 'Neustadt a.d.Waldnaab',
                                    "Fürstenfeldbruck", 'Rosenheim', "Straubing", "Erding", "Tirschenreuth", 'Miltenberg', 'Neumarkt i.d.OPf.']}

# colors for curves
C1 = "#D55E00"
C2 = "#E69F00"
C3 = "#0073CF"

with open('../data/comparison.pkl',"rb") as f:
    best_model=pkl.load(f)

for i,disease in enumerate(diseases):
    # Load data
    use_age = best_model[disease]["use_age"]
    use_eastwest = best_model[disease]["use_eastwest"]
    if disease=="borreliosis":
        prediction_region = "bavaria"
        use_eastwest = False
    else:
        prediction_region = "germany"
        
    data = load_data(disease, prediction_region, counties)
    data = data[data.index < parse_yearweek("2018-KW1")]
    if disease == "borreliosis":
        data = data[data.index >= parse_yearweek("2013-KW1")]
    _, _, _, target = split_data(data)
    county_ids = target.columns

    # Load our prediction samples
    res = load_pred(disease, use_age, use_eastwest)

    prediction_samples = np.reshape(res['y'],(res["y"].shape[0],104,-1))
    prediction_quantiles = quantiles(prediction_samples,(5,25,75,95))

    prediction_mean = pd.DataFrame(data=np.mean(prediction_samples,axis=0), index=target.index, columns=target.columns)
    prediction_q25 = pd.DataFrame(data=prediction_quantiles[25], index=target.index, columns=target.columns)
    prediction_q75 = pd.DataFrame(data=prediction_quantiles[75], index=target.index, columns=target.columns)
    prediction_q5 = pd.DataFrame(data=prediction_quantiles[5], index=target.index, columns=target.columns)
    prediction_q95 = pd.DataFrame(data=prediction_quantiles[95], index=target.index, columns=target.columns)

    # Load hhh4 predictions for reference
    hhh4_predictions = pd.read_csv("../data/diseases/{}_hhh4.csv".format("borreliosis_notrend" if disease=="borreliosis" else disease))
    weeks = hhh4_predictions.pop("weeks")
    hhh4_predictions.index = parse_yearweek(weeks)

    fig = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(5, 5, top=0.90, bottom=0.11, left=0.07, right=0.92, hspace=0.2, wspace=0.3)

    for j,name in enumerate(plot_county_names[disease]):

        ax = fig.add_subplot(grid[np.unravel_index(list(range(25))[j],(5,5))])

        county_id = countyByName[name]
        dates = [n.wednesday() for n in target.index.values]

        # plot our predictions w/ quartiles
        p_pred=ax.plot_date(dates, prediction_mean[county_id], "-", color=C1, linewidth=2.0, zorder=4)
        p_quant=ax.fill_between(dates, prediction_q25[county_id], prediction_q75[county_id], facecolor=C2, alpha=0.5, zorder=1)
        ax.plot_date(dates, prediction_q25[county_id], ":", color=C2, linewidth=2.0, zorder=3)
        ax.plot_date(dates, prediction_q75[county_id], ":", color=C2, linewidth=2.0, zorder=3)

        # plot hhh4 reference prediction
        p_hhh4=ax.plot_date(dates, hhh4_predictions[county_id], "-", color=C3, linewidth=2.0, zorder=3)

        # plot ground truth
        p_real=ax.plot_date(dates, target[county_id], "k.")

        ax.set_title(name, fontsize=18)
        ax.tick_params(axis="both", direction='out', size=2, labelsize=14)
        plt.setp(ax.get_xticklabels(), visible=j>19, rotation=60)

        ax.autoscale(False)
        p_quant2=ax.fill_between(dates, prediction_q5[county_id], prediction_q95[county_id], facecolor=C2, alpha=0.25, zorder=0)
        ax.plot_date(dates, prediction_q5[county_id], ":", color=C2, alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot_date(dates, prediction_q95[county_id], ":", color=C2, alpha=0.5, linewidth=2.0, zorder=1)

    plt.legend([p_real[0], p_pred[0], p_hhh4[0], p_quant, p_quant2],
    ["reported", "predicted", "hhh4", "25\%-75\% quantile", "5\%-95\% quantile"],
    fontsize=16, ncol=5, loc="upper center", bbox_to_anchor = (0,-0.01,1,1),
        bbox_transform = plt.gcf().transFigure )
    fig.text(0.5, 0.02, "Time [calendar weeks]", ha='center', fontsize=22)
    fig.text(0.01, 0.46, "Reported/predicted infections", va='center', rotation='vertical', fontsize=22)
    plt.savefig("../figures/curves_{}_appendix.pdf".format(disease))

# plt.show()
