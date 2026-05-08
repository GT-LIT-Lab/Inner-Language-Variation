# Utility functions for simulating data

import os
from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root_scalar
from scipy.special import expit
from tqdm import tqdm


DEFAULT_R_BACKEND_COMMAND = "apptainer"
DEFAULT_R_APPTAINER_PATH = (
    "/usr/local/pace-apps/manual/packages/singularity_images/"
    "r-4.2.1-rocker.sif"
)


def generate_correlated_effects(n: int,
                                sd_values: Union[list[int], list[float], np.ndarray],
                                corr_matrix: Union[np.ndarray, list[list]],
                                method: str = "cholesky"):
    """
    Generates correlated random effects using a multivariate normal distribution 
    and Cholesky decomposition or eigen-value decomposition.

    Adapted from Will's wiscs code
    will.decker@gatech.edu

    Parameters
    ----------
    n:int 
    sd_values: list
    corr_matrix: Union[np.ndarray, list[list]].
    method: str, default="cholesky"
    
    Return
    -------
    np.ndarray: A (n, len(sd_values)) array of random effects.
    """
    sd_values = np.asarray(sd_values, dtype=float)
    k = len(sd_values)
    if sd_values.ndim != 1:
        raise ValueError("sd_values must be 1D")

    # construct variance-covariance matrix
    cov_matrix = np.diag(sd_values) @ np.array(corr_matrix) @ np.diag(sd_values)

    # decompose with Cholesky or eigen-value decomposition
    if method == "cholesky":
        L = np.linalg.cholesky(cov_matrix)
    elif method == "eigen":
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals[eigvals < 1e-12] = 0 # zero out negative eigenvals
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    else:
        raise ValueError("Unknown method. Must be 'cholesky' or 'eigen'.")

    z = np.random.randn(n, k)  # independent normal
    return z @ L.T


def configure_r_backend(
    backend,
    command: str = DEFAULT_R_BACKEND_COMMAND,
    apptainer_path: str = DEFAULT_R_APPTAINER_PATH,
):
    backend.command = command
    backend.apptainer_path = apptainer_path


def select_array_job(*search_spaces):
    """
    Map ``SLURM_ARRAY_TASK_ID`` onto a Cartesian product of parameter grids.

    When the environment variable is absent, default to the first job so the
    scripts remain importable and easier to run locally.
    """
    array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    sizes = [len(space) for space in search_spaces]
    total_jobs = int(np.prod(sizes))

    if not 0 <= array_task_id < total_jobs:
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={array_task_id} is out of range for "
            f"{total_jobs} jobs."
        )

    selected = []
    remainder = array_task_id
    for i, space in enumerate(search_spaces):
        block_size = int(np.prod(sizes[i + 1:])) if i + 1 < len(sizes) else 1
        selected_index = remainder // block_size
        remainder %= block_size
        selected.append(space[selected_index])

    return tuple(selected)


def interior_linspace(lower, upper, n_points):
    return np.linspace(lower, upper, n_points)[1:-1]


def logistic(z):
    return expit(z)


def logistic_squared(z):
    return logistic(z) ** 2

# -------- Gauss–Hermite quadrature (M = 64 nodes) -------
def gh_expect(mu, sig, func):
    M = 64
    nodes, weights = hermgauss(M)
    z = np.sqrt(2) * sig * nodes + mu
    return (weights @ func(z)) / np.sqrt(np.pi)

def meanY(mu, sig):
    return gh_expect(mu, sig, logistic)

def varY(mu, sig):
    return gh_expect(mu, sig, logistic_squared) - meanY(mu, sig) ** 2

# ---------------- robust inner root: μ ------------------
def choose_mu(p, sig, xtol=1e-12):
    """solve meanY(μ,σ)=p  using Brent with adaptive bracket."""
    kappa = np.sqrt(1 + np.pi**2 / 3)
    centre = kappa * stats.norm.ppf(p)
    width = max(30 * sig, 5.0)             # absolute width floor

    f = lambda mu: meanY(mu, sig) - p
    for _ in range(15):                    # widen until sign flips
        lo, hi = centre - width, centre + width
        if np.sign(f(lo)) * np.sign(f(hi)) < 0:
            return root_scalar(f, bracket=[lo, hi],
                                method="brentq", xtol=xtol).root
        width *= 2
    raise RuntimeError("choose_mu: could not bracket root.")

# ---------------- robust outer root: σ ------------------
def choose_sigma(p, target_sd,
                 lo=1e-6, hi=5.0, max_hi=100, xtol=1e-12):
    sd_cap = np.sqrt(p * (1 - p))          # theoretical maximum
    if target_sd >= sd_cap:
        raise ValueError("target_sd ≥ √(p(1-p)); no solution.")

    def f(sig):
        return varY(choose_mu(p, sig), sig) - target_sd**2

    if f(lo) > 0:
        lo *= 0.1                          # ensure f(lo) < 0
    while f(hi) < 0 and hi < max_hi:
        hi *= 2                            # expand until sign flip
    if hi >= max_hi and f(hi) < 0:
        raise RuntimeError("choose_sigma: could not bracket root.")

    return root_scalar(f, bracket=[lo, hi],
                       method="brentq", xtol=xtol).root

def solve_mu_sigma(p, sY):
    sigma = choose_sigma(p, sY)
    mu = choose_mu(p, sigma)
    return mu, sigma

# ---------- recover (μ,σ) from binomial mean & SD -------
def mu_sigma_from_binom(mean_X, sd_X, n):
    p = mean_X / n
    v = (sd_X**2 - n * p * (1 - p)) / (n**2 - n)
    if not (0 < p < 1) or v < 0 or v > p*(1-p):
        raise ValueError("mean/SD combo impossible for logistic-normal.")
    mu, sigma = solve_mu_sigma(p, np.sqrt(v))
    return mu, sigma, p, np.sqrt(v)


def simulate(n_subjects, n_trials, p, std, corr, modalities, scenarios, 
             stats_func, n_boot=1000, agg=True):
    """
    Simulate data for a given number of subjects, trials, and modalities.

    Parameters
    ----------
    n_subjects: int
    n_trials: int
    p: list
        Probability of success for each modality.
    std: list
        Standard deviation for each modality.
    corr: list
        Correlation matrix for the modalities. A 2D list or numpy array.
    modalities: list
        List of modality names.
    scenarios: dict
        Dictionary of scenarios with modality effects.
    stats_func: function
        Function to compute statistics on the simulated data.
    n_boot: int
        Number of bootstrap samples to generate.

    Returns
    -------
    stats_data: list
        List of statistics computed from the simulated data.
    """
    np.random.seed(0)

    subjects = np.random.choice(range(10000, 100000), n_subjects, replace=False)

    mu = []
    sigma = []
    for i in range(len(p)):
        mu_i, sigma_i, _, _ = mu_sigma_from_binom(
            p[i] * n_trials, std[i], n_trials
        )
        mu.append(mu_i)
        sigma.append(sigma_i)

    modality_to_index = {
        modality: modality_i for modality_i, modality in enumerate(modalities)
    }
    stats_data = []
    for _ in tqdm(range(n_boot)):
        latent_main = generate_correlated_effects(n_subjects, sigma, corr)
        for modality_i in range(len(modalities)):
            latent_main[:, modality_i] = latent_main[:, modality_i] + mu[modality_i]

        latent_main_df = pd.DataFrame(latent_main, columns=modalities)
        latent_main_df["subject"] = subjects

        latent_main_df = pd.concat([latent_main_df] * n_trials, ignore_index=True)
        latent_main_df['trial'] = np.repeat(range(1, n_trials + 1), n_subjects)

        binary_data = []
        for scenario, seffect, in scenarios.items():
            binary_df = pd.DataFrame({
                'subject': latent_main_df['subject'], 
                'trial': latent_main_df['trial']})
            binary_df['scenario'] = scenario
            for modality in modalities:
                modality_i = modality_to_index[modality]
                new_mu = choose_mu(
                    seffect.get(modality, 0) + p[modality_i],
                    sigma[modality_i],
                )
                seff = new_mu - mu[modality_i]

                logit_data = expit(latent_main_df[modality] + seff)
                binary_df[modality] = np.random.binomial(1, logit_data)
            binary_data.append(binary_df)
        binary_data = pd.concat(binary_data, ignore_index=True)

        binary_data = pd.melt(
            binary_data,
            id_vars=['subject', 'trial', 'scenario'],
            value_vars=modalities,
            var_name='modality',
            value_name='response'
        )
        binary_data['response'] = binary_data['response'].astype(int)

        if agg:
            binary_data = binary_data[
                ['subject', 'modality', 'scenario', 'response']
            ].groupby(
                ['subject', 'modality', 'scenario']
            ).mean().reset_index()

        results = stats_func(binary_data)
        stats_data.append(results)

    return stats_data

def calculate_power(data0, data, threshold=0.95):
    critical_value = np.quantile(data0, threshold)
    power = np.mean(data > critical_value)
    return power

def check_compatibility(p, sd, n):
    lower = np.sqrt(n * p * (1 - p)) / n
    upper = np.sqrt(n * p * (1 - p) + (n**2 - n) * p * (1 - p)) / n
    return lower <= sd <= upper, lower, upper


def theoretical_binomial_corr(mu, sigma, rho, n_trials=1, M=64):
    """
    Compute theoretical correlation in the binomial space given latent Gaussian with logit link.

    Parameters:
        mu: list of length 2, means of latent Gaussian variables
        sigma: list of length 2, std devs
        rho: float, latent correlation
        n_trials: int, number of binomial trials
        M: int, quadrature resolution

    Returns:
        rho_Y: float, correlation of observed binomial variables
    """
    nodes, weights = hermgauss(M)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)

    x, y = np.meshgrid(nodes, nodes)
    wx, wy = np.meshgrid(weights, weights)
    w2d = wx * wy

    # Bivariate Gaussian transform
    z1 = sigma[0] * x + mu[0]
    z2 = sigma[1] * (rho * x + np.sqrt(1 - rho**2) * y) + mu[1]

    # Logit-inverse
    p1 = 1 / (1 + np.exp(-z1))
    p2 = 1 / (1 + np.exp(-z2))

    # Expectations
    Ep1 = np.sum(p1 * w2d)
    Ep2 = np.sum(p2 * w2d)
    Ep1p2 = np.sum(p1 * p2 * w2d)
    Ep1sq = np.sum(p1**2 * w2d)
    Ep2sq = np.sum(p2**2 * w2d)
    Ep1_one_minus_p1 = np.sum(p1 * (1 - p1) * w2d)
    Ep2_one_minus_p2 = np.sum(p2 * (1 - p2) * w2d)

    cov_p = Ep1p2 - Ep1 * Ep2
    var_p1 = Ep1sq - Ep1**2
    var_p2 = Ep2sq - Ep2**2

    # Final numerator and denominator
    numerator = n_trials**2 * cov_p
    denom = np.sqrt(
        (n_trials * Ep1_one_minus_p1 + n_trials**2 * var_p1) *
        (n_trials * Ep2_one_minus_p2 + n_trials**2 * var_p2)
    )

    return numerator / denom
