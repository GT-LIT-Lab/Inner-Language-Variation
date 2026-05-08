import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats

from utils import (
    calculate_power,
    check_compatibility,
    interior_linspace,
    mu_sigma_from_binom,
    simulate,
    theoretical_binomial_corr,
)


n_subjects_search = [100, 500, 1000]
n_trials_search = [2, 5, 10]
corr_search = [0.1, 0.2, 0.4]
p_search = [0.1, 0.5, 0.9]
MODALITIES = ["orthographic", "auditory"]
SCENARIOS = {"spontaneous": {}}


def compute_corr(x):
    return stats.pearsonr(
        x[x["modality"] == "auditory"]["response"],
        x[x["modality"] == "orthographic"]["response"],
    )


simulated = []
for n_subjects in n_subjects_search:
    for n_trials in n_trials_search:
        for corr in corr_search:
            for p0, p1 in itertools.product(p_search, p_search):
                p = [p0, p1]
                _, lower1, upper1 = check_compatibility(p[0], 0, n_trials)
                _, lower2, upper2 = check_compatibility(p[1], 0, n_trials)
                sd_list1 = interior_linspace(lower1, upper1, 5)
                sd_list2 = interior_linspace(lower2, upper2, 5)

                for sd1, sd2 in itertools.product(sd_list1, sd_list2):
                    std = [sd1 * n_trials, sd2 * n_trials]

                    mu1, sigma1, _, _ = mu_sigma_from_binom(
                        p[0] * n_trials, std[0], n_trials
                    )
                    mu2, sigma2, _, _ = mu_sigma_from_binom(
                        p[1] * n_trials, std[1], n_trials
                    )
                    rho_y = theoretical_binomial_corr(
                        [mu1, mu2], [sigma1, sigma2], corr, n_trials=n_trials
                    )

                    data0 = simulate(
                        n_subjects=n_subjects,
                        n_trials=n_trials,
                        p=p,
                        std=std,
                        corr=[[1, 0.], [0., 1]],
                        modalities=MODALITIES,
                        scenarios=SCENARIOS,
                        stats_func=compute_corr,
                    )

                    data = simulate(
                        n_subjects=n_subjects,
                        n_trials=n_trials,
                        p=p,
                        std=std,
                        corr=[[1, corr], [corr, 1]],
                        modalities=MODALITIES,
                        scenarios=SCENARIOS,
                        stats_func=compute_corr,
                    )

                    power = calculate_power(data0, data, threshold=0.95)
                    simulated.append({
                        'n_subjects': n_subjects,
                        'n_trials': n_trials,
                        'corr': corr,
                        'p0': p[0],
                        'p1': p[1],
                        'sd1': sd1,
                        'sd2': sd2,
                        'power': power,
                        # theoretical values
                        'lower1': lower1,
                        'upper1': upper1,
                        'lower2': lower2,
                        'upper2': upper2,
                        'mu1': mu1,
                        'mu2': mu2,
                        'sigma1': sigma1,
                        'sigma2': sigma2,
                        'rho_y': rho_y,
                    })
                    print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, "
                          f"corr: {corr}, p0: {p[0]}, p1: {p[1]}, "
                          f"sd1: {sd1}, sd2: {sd2}, power: {power}")
                    
        pd.DataFrame(simulated).to_csv('simulate4power_data/corr.csv', index=False)
