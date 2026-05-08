import itertools
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
dp_search = [-0.01, -0.05, -0.1]
p_search = [0.2, 0.5, 0.8]
MODALITIES = ["auditory", "orthographic"]
SCENARIOS = {"spontaneous": {}}

def compute_corr(x):
    code = f'''
    long_df <- {to_r(x)}
    ''' +'''

    library(tidyr, quietly = T)
    library(dplyr, quietly = T) 
    library(lmerTest, quietly = T)
    library(emmeans, quietly = T)

    long_df$modality <- factor(long_df$modality, levels=c('auditory', 'orthographic'))

    model <- glmer(
        response ~ 1 + modality + (1 + modality| subject),
        data = long_df,
        family = binomial(link = "logit"),
        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5))
    )

    print(summary(model))

    z <- summary(model)$coefficients['modalityorthographic', 'z value']

    # @grab{float}
    z
    '''
    try:
        return R(code, grab=True)
    except Exception as e:
        print(f"Error: {e}")
        return np.nan

n_subjects, n_trials, dp = select_array_job(
    n_subjects_search,
    n_trials_search,
    dp_search,
)
print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, dp: {dp}")

simulated = []

for p0 in p_search:
    p = [p0, p0 + dp]
    p1 = p[1]

    _, lower1, upper1 = check_compatibility(p0, 0, n_trials)
    sd_list1 = interior_linspace(lower1, upper1, 4)
    _, lower2, upper2 = check_compatibility(p1, 0, n_trials)
    sd_list2 = interior_linspace(lower2, upper2, 4)

    for sd1, sd2 in itertools.product(sd_list1, sd_list2):
        std = [sd1 * n_trials, sd2 * n_trials]

        mu1, sigma1, _, _ = mu_sigma_from_binom(
            p0 * n_trials, sd1 * n_trials, n_trials
        )
        mu2, sigma2, _, _ = mu_sigma_from_binom(
            p1 * n_trials, sd2 * n_trials, n_trials
        )

        data0 = simulate(
            n_subjects=n_subjects,
            n_trials=n_trials,
            p=[p0, p0],
            std=std,
            corr=[[1, 0.], [0., 1]],
            modalities=MODALITIES,
            scenarios=SCENARIOS,
            stats_func=compute_corr,
            agg=False,
        )

        data = simulate(
            n_subjects=n_subjects,
            n_trials=n_trials,
            p=p,
            std=std,
            corr=[[1, 0.], [0., 1]],
            modalities=MODALITIES,
            scenarios=SCENARIOS,
            stats_func=compute_corr,
            agg=False,
        )


        power = calculate_power(data0, data, threshold=0.95)

        simulated.append({
            'n_subjects': n_subjects,
            'n_trials': n_trials,
            'p1': p[0],
            'sd1': std[0],
            'p2': p[1],
            'sd2': std[1],
            'power': power,
            # theoretical values
            'lower1': lower1,
            'upper1': upper1,
            'lower2': lower2,
            'upper2': upper2,
            'mu1': mu1,
            'sigma1': sigma1,
            'mu2': mu2,
            'sigma2': sigma2,
        })
        print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, power: {power}")
            
        pd.DataFrame(simulated).to_csv(f'simulate4power_data/dp_s{n_subjects}t{n_trials}dp{dp}.csv', index=False)
