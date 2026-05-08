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
deffect_search = [0.01, 0.05, 0.1]
p_search = [0.2, 0.5, 0.8]
MODALITIES = ["auditory"]
BASELINE_SCENARIOS = {
    "spontaneous": {},
    "inner_hearing_general": {"auditory": 0.0},
}

def compute_scenario_effect(x):
    code = f'''
    long_df <- {to_r(x)}
    ''' +'''

    library(dplyr, quietly = T) 
    library(lmerTest, quietly = T)

    long_df$scenario <- factor(long_df$scenario, levels=c('spontaneous', 'inner_hearing_general'))

    model <- glmer(
        response ~ 1 + scenario + (1 | subject),
        data = long_df,
        family = binomial(link = "logit"),
        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5))
    )

    z <- summary(model)$coefficients['scenarioinner_hearing_general', 'z value']

    # @grab{float}
    z
    '''

    try:
        res = R(code, grab=True)
        return res
    except Exception as e:
        print(f"Error in R code execution: {e}")
        return np.nan

n_subjects, n_trials, deffect = select_array_job(
    n_subjects_search,
    n_trials_search,
    deffect_search,
)
print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, deffect: {deffect}")

simulated = []

for p0 in p_search:
    p = [p0]
    _, lower, upper = check_compatibility(p0, 0, n_trials)
    sd_list = interior_linspace(lower, upper, 4)

    for sd in sd_list:
        std = [sd * n_trials]

        mu, sigma, _, _ = mu_sigma_from_binom(
            p0 * n_trials, sd * n_trials, n_trials
        )


        data0 = simulate(
            n_subjects=n_subjects,
            n_trials=n_trials,
            p=p,
            std=std,
            corr=[[1]],
            modalities=MODALITIES,
            scenarios=BASELINE_SCENARIOS,
            stats_func=compute_scenario_effect,
            agg=False,
        )

        data = simulate(
            n_subjects=n_subjects,
            n_trials=n_trials,
            p=p,
            std=std,
            corr=[[1]],
            modalities=MODALITIES,
            scenarios={
                "spontaneous": {},
                "inner_hearing_general": {"auditory": deffect},
            },
            stats_func=compute_scenario_effect,
            agg=False,
        )

        power = calculate_power(data0, data, threshold=0.95)

        simulated.append({
            'n_subjects': n_subjects,
            'n_trials': n_trials,
            'deffect': deffect,
            'p': p[0],
            'sd': std[0],
            'power': power,
            'lower': lower,
            'upper': upper,
            'mu': mu,
            'sigma': sigma
        })
        print(f"n_subjects: {n_subjects}, n_trials: {n_trials}, "
                f"sd: {sd}, p0: {p0}, power: {power}")
            
        pd.DataFrame(simulated).to_csv(f'simulate4power_data/scenario_s{n_subjects}t{n_trials}e{deffect}.csv', index=False)
