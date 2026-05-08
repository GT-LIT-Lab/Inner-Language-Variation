import numpy as np
import pandas as pd
import rinterface.rinterface as R
import rinterface.backend as bk
from rinterface.utils import to_r

from utils import (
    calculate_power,
    check_compatibility,
    configure_r_backend,
    interior_linspace,
    mu_sigma_from_binom,
    select_array_job,
    simulate,
)

configure_r_backend(bk)

n_subjects_search = [100, 500, 1000]
n_trials_search = [5, 10, 15]
p_search = [0.51, 0.55, 0.6]
MODALITIES = ["auditory"]
SCENARIOS = {"spontaneous": {}}

def compute_effect(x):
    code = f'''
    long_df <- {to_r(x)}
    ''' +'''

    library(tidyr, quietly = T)
    library(dplyr, quietly = T) 
    library(lmerTest, quietly = T)

    model <- glmer(
        response ~ 1 + (1 | subject),
        data = long_df,
        family = binomial(link = "logit"),
        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5))
    )

    z <- summary(model)$coefficients['(Intercept)', 'z value']

    # @grab{float}
    z
    '''

    try:
        res = R(code, grab=True)
        return res
    except Exception as e:
        print(f"Error in R code execution: {e}")
        return np.nan


n_subjects, n_trials, p0 = select_array_job(
    n_subjects_search,
    n_trials_search,
    p_search,
)
print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, p: {p0}")

simulated = []

p = [p0]
_, lower0, upper0 = check_compatibility(0.5, 0, n_trials)
_, lower1, upper1 = check_compatibility(p0, 0, n_trials)
lower = max(lower0, lower1)
upper = min(upper0, upper1)
sd_list = interior_linspace(lower, upper, 4)

for sd in sd_list:
    std = [sd * n_trials]

    mu, sigma, _, _ = mu_sigma_from_binom(
        p0 * n_trials, sd * n_trials, n_trials
    )


    data0 = simulate(
        n_subjects=n_subjects,
        n_trials=n_trials,
        p=[0.5],
        std=std,
        corr=[[1]],
        modalities=MODALITIES,
        scenarios=SCENARIOS,
        stats_func=compute_effect,
        agg=False,
    )

    data = simulate(
        n_subjects=n_subjects,
        n_trials=n_trials,
        p=p,
        std=std,
        corr=[[1]],
        modalities=MODALITIES,
        scenarios=SCENARIOS,
        stats_func=compute_effect,
        agg=False,
    )

    power = calculate_power(data0, data, threshold=0.95)

    simulated.append({
        'n_subjects': n_subjects,
        'n_trials': n_trials,
        'p': p[0],
        'sd': std[0],
        'power': power,
        # theoretical values
        'lower': lower,
        'upper': upper,
        'mu': mu,
        'sigma': sigma
    })
    print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, "
            f"sd: {sd}, p0: {p0}, power: {power}")
        
    pd.DataFrame(simulated).to_csv(f'simulate4power_data/p50_s{n_subjects}t{n_trials}p{p0}.csv', index=False)
