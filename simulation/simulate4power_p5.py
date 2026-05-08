import numpy as np
import pandas as pd
import scipy.stats as stats

from utils import (
    calculate_power,
    check_compatibility,
    interior_linspace,
    mu_sigma_from_binom,
    simulate,
)

n_subjects_search = [100, 500, 1000]
n_trials_search = [5, 10, 15]
p_search = [0.06, 0.10, 0.15]
MODALITIES = ["auditory"]
SCENARIOS = {"spontaneous": {}}


def wilcoxon_stat(x):
    return stats.wilcoxon(
        x["response"] - 0.05,
        alternative="greater",
    ).statistic


simulated = []
for n_subjects in n_subjects_search:
    for n_trials in n_trials_search:
        for p0 in p_search:
            p = [p0]
            _, lower1, upper1 = check_compatibility(p0, 0, n_trials)
            _, lower2, upper2 = check_compatibility(0.05, 0, n_trials)
            lower = max(lower1, lower2)
            upper = min(upper1, upper2)

            sd_list = interior_linspace(lower, upper, 5)

            for sd in sd_list:
                try:
                    std = [sd * n_trials]

                    mu, sigma, _, _ = mu_sigma_from_binom(
                        p[0] * n_trials, std[0], n_trials
                    )

                    data0 = simulate(
                        n_subjects=n_subjects,
                        n_trials=n_trials,
                        p=[.05],
                        std=std,
                        corr=[[1]],
                        modalities=MODALITIES,
                        scenarios=SCENARIOS,
                        stats_func=wilcoxon_stat,
                        agg=True,
                    )

                    data = simulate(
                        n_subjects=n_subjects,
                        n_trials=n_trials,
                        p=p,
                        std=std,
                        corr=[[1]],
                        modalities=MODALITIES,
                        scenarios=SCENARIOS,
                        stats_func=wilcoxon_stat,
                        agg=True,
                    )

                    power = calculate_power(data0, data, threshold=0.95)
                    simulated.append({
                        'n_subjects': n_subjects,
                        'n_trials': n_trials,
                        'p': p0,
                        'sd': sd,
                        'power': power,
                        # theoretical values
                        'lower': lower,
                        'upper': upper,
                        'mu': mu,
                        'sigma': sigma
                    })
                    print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, "
                            f"p={p0}, sd={sd}, power: {power}")
                    
                    
                    pd.DataFrame(simulated).to_csv('simulate4power_data/p5_2.csv', index=False)
                except ValueError as e:
                    print(f"Error: {e}")
                    continue
