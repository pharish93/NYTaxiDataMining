import numpy as np
from scipy.stats import chi2

from matplotlib import pyplot as plt

from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def likelihood_ratio(ll_global, ll_region, ll_not_region):
    return(2*(ll_global-ll_region-ll_not_region))

def lrt_taxi_data(train_data, train_durations, regions):
    gp_global = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))

    p_values_all_regions = dict()

    gp_global.fit(train_data, train_durations)
    ll_global = gp_global.log_marginal_likelihood(gp_global.kernel_.theta)

    for region in regions:
        gp_region = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        gp_not_region = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0))
        gp_region.fit(train_data.loc[train_data['region'] == region])
        gp_not_region.fit(train_data.loc[train_data['region'] != region])
        ll_region = gp_region.log_marginal_likelihood(gp_region.kernel_.theta)
        ll_not_region = gp_not_region.log_marginal_likelihood(gp_not_region.kernel_.theta)

        region_lrt = likelihood_ratio(ll_global, ll_region, ll_not_region)
        df = 1

        p = chi2.cdf(region_lrt, df)
        p_values_all_regions[region] = p

    return p_values_all_regions
