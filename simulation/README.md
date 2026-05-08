# Simulation Cleanup

This folder contains a compact set of scripts for power simulations based on a
logistic-normal subject model.

## Files

- `utils.py`: shared helpers for latent-variable simulation, power calculation,
  compatibility checks, theoretical correlation, SLURM array selection, and R
  backend setup.
- `simulate4power_corr.py`: power simulation for detecting cross-modality
  correlation changes.
- `simulate4power_dp.py`: power simulation for modality differences in baseline
  probability (`dp`), evaluated with an R mixed-effects model.
- `simulate4power_scenario.py`: power simulation for scenario effects such as
  `inner_hearing_general`.
- `simulate4power_p50.py`: power simulation for testing probabilities above a
  0.50 reference level with an R mixed-effects model.
- `simulate4power_p5.py`: power simulation for testing probabilities above a
  0.05 reference level with a Wilcoxon-based statistic.

## Prerequisite

The R-based simulation scripts require `rinterface` as a prerequisite:
[w-decker/rinterface](https://github.com/w-decker/rinterface).
