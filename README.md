# The BSTI Model

The BSTI model is a [Bayesian spatio-temporal interaction model][1], a probabilistic generalized linear model, that predicts aggregated case counts within spatial regions (counties) and time intervals (calendar weeks) using a history of reported cases, temporal features (seasonality and trend) and region-specific as well as demographic information.

The model is implemented in Python and relies on the [PyMC3][2] and [Theano][3] packages for computationally efficient sampling.

Key features of the model:

* a single probabilistic model learns to predict the number of weekly case counts for three different diseases (campylobacteriosis, rotaviralenteritis and Lyme borreliosis) at the county level one week ahead of time
* a Bayesian Monte Carlo regression approach provides an estimate of the full probability distribution over inferred parameters as well as model predictions.
* the model learns an interpretable spatio-temporal kernel that captures typical interactions between infection cases of the tested diseases.

# Data sources
## Epidemiological data
The data is provided by the [Robert Koch Institute][4], and consists of weekly reports of case counts for three diseases, campylobacteriosis, rotavirus infections and Lyme borreliosis. They are aggregated by county and collected over a time period spanning from the 1st of January 2011 (2013 for borreliosis) to the 31st of December 2017 via the *SurvNet* surveillance system. Aggregated case counts of diseases with mandatory reporting in Germany is available [online][5].

## Geospatial data
Information about the shape of counties within Germany is publicly provided by the [German federal agency for cartography and geodesy (Bundesamt für Kartographie und Geodäsie)][6] (© GeoBasis-DE / BKG 2018) under the [dl-de/by-2-0 license][7].


[1]: https://www.biorxiv.org/content/10.1101/617795v1
[2]: https://docs.pymc.io/
[3]: https://github.com/Theano/Theano
[4]: https://rki.de
[5]: https://survstat.rki.de
[6]: http://www.bkg.bund.de
[7]: https://www.govdata.de/dl-de/by-2-0
