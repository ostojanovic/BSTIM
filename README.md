# The BSTI Model

The BSTI model is a Bayesian spatio-temporal interaction model, a probabilistic generalized linear model, that predicts aggregated case counts within spatial regions (counties) and time intervals (calendar weeks) using a history of reported cases, temporal features (seasonality and trend) and region-specific as well as demographic information.

The model is implemented in Python and relies on the [PyMC3][1] and [Theano][2] packages for computationally efficient sampling.

The data is provided by the [Robert Koch Institute][3], and consists of weekly reports of case counts for three diseases, campylobacteriosis, rotavirus infections and Lyme borreliosis. They are aggregated by county and collected over a time period spanning from the 1st of January 2011 (2013 for borreliosis) to the 31st of December 2017 via the *SurvNet* surveillance system. Aggregated case counts of diseases with mandatory reporting in Germany is available [online][4].

Key features of the model:

* a single probabilistic model learns to predict the number of weekly case counts for three different diseases (campylobacteriosis, rotaviralenteritis and Lyme borreliosis) at the county level one week ahead of time
* a Bayesian Monte Carlo regression approach provides an estimate of the full probability distribution over inferred parameters as well as model predictions.
* the model learns an interpretable spatio-temporal kernel that captures typical interactions between infection cases of the tested diseases.



[1]: https://docs.pymc.io/
[2]: https://github.com/Theano/Theano
[3]: https://rki.de
[4]: https://survstat.rki.de
