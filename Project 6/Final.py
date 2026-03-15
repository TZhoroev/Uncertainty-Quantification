"""Dittus-Boelter model calibration, Bayesian inference, and diagnostics.

The model is

    Nu = theta1 * Re**theta2 * Pr**theta3

This script reproduces the major MATLAB workflow in Python:
1. load Reynolds, Prandtl, and Nusselt data;
2. compute residual diagnostics for the initial guess;
3. estimate parameters with nonlinear least squares;
4. assemble the sensitivity matrix and Fisher information matrix;
5. run a Bayesian Metropolis-within-Gibbs MCMC scheme;
6. generate residual, chain, density, pairwise, and prediction-interval plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde, invgamma


INITIAL_PARAMS = np.array([0.023, 0.8, 0.4], dtype=float)
MCMC_SAMPLES = 20000
MCMC_BURN_IN = 5000
MCMC_SEED = 123
IDENTIFIABILITY_TOL = 1.0e-8


def load_db_data(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Reynolds, Prandtl, and Nusselt data from the text file."""
    data = np.loadtxt(file_path, comments="%")
    return data[:, 0], data[:, 1], data[:, 2]


def dittus_boelter(params: np.ndarray, re_data: np.ndarray, pr_data: np.ndarray) -> np.ndarray:
    """Evaluate the Dittus-Boelter correlation model."""
    theta_1, theta_2, theta_3 = params
    return theta_1 * (re_data ** theta_2) * (pr_data ** theta_3)


def residuals(params: np.ndarray, re_data: np.ndarray, pr_data: np.ndarray, nu_data: np.ndarray) -> np.ndarray:
    """Return model residuals ``Nu_model - Nu_data``."""
    return dittus_boelter(params, re_data, pr_data) - nu_data


def sensitivity_matrix(params: np.ndarray, re_data: np.ndarray, pr_data: np.ndarray) -> np.ndarray:
    """Construct the model sensitivity matrix with respect to the parameters."""
    theta_1, theta_2, theta_3 = params
    base = (re_data ** theta_2) * (pr_data ** theta_3)
    return np.column_stack(
        [
            base,
            theta_1 * base * np.log(re_data),
            theta_1 * base * np.log(pr_data),
        ]
    )


def pss_svd(sens_mat: np.ndarray, eta: float) -> tuple[list[int], list[int]]:
    """Identify practically identifiable parameters using the MATLAB-style SVD test."""
    identifiable = list(range(sens_mat.shape[1]))
    working_matrix = sens_mat.copy()

    while working_matrix.shape[1] > 0:
        _, singular_values, vh = np.linalg.svd(working_matrix, full_matrices=False)
        if singular_values[-1] / singular_values[0] > eta:
            break
        remove_idx = int(np.argmax(np.abs(vh[-1])))
        del identifiable[remove_idx]
        working_matrix = np.delete(working_matrix, remove_idx, axis=1)

    unidentifiable = [idx for idx in range(sens_mat.shape[1]) if idx not in identifiable]
    return [idx + 1 for idx in identifiable], [idx + 1 for idx in unidentifiable]


def estimate_parameters(
    initial_params: np.ndarray,
    re_data: np.ndarray,
    pr_data: np.ndarray,
    nu_data: np.ndarray,
) -> least_squares:
    """Estimate model parameters with nonlinear least squares."""
    return least_squares(
        residuals,
        x0=initial_params,
        args=(re_data, pr_data, nu_data),
        bounds=([1.0e-8, 0.0, 0.0], [1.0, 2.0, 2.0]),
        method="trf",
        jac="2-point",
    )


def log_prior(theta: np.ndarray) -> float:
    """Weakly informative prior enforcing positive parameters."""
    if np.any(theta <= 0.0):
        return -np.inf
    prior_mean = np.array([0.02, 0.8, 0.4])
    prior_std = np.array([0.02, 0.4, 0.3])
    z = (theta - prior_mean) / prior_std
    return -0.5 * np.dot(z, z)


def sse(theta: np.ndarray, re_data: np.ndarray, pr_data: np.ndarray, nu_data: np.ndarray) -> float:
    """Compute the sum of squared residuals."""
    res = residuals(theta, re_data, pr_data, nu_data)
    return float(res @ res)


def run_mcmc(
    theta_init: np.ndarray,
    sigma2_init: float,
    proposal_cov: np.ndarray,
    re_data: np.ndarray,
    pr_data: np.ndarray,
    nu_data: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run a Metropolis-within-Gibbs sampler for theta and sigma^2."""
    rng = np.random.default_rng(seed)
    n_data = len(nu_data)
    n_params = len(theta_init)
    stabilised_cov = proposal_cov + 1.0e-12 * np.eye(n_params)
    proposal_scale = (2.38 ** 2) / n_params
    proposal_chol = np.linalg.cholesky(proposal_scale * stabilised_cov)

    theta_chain = np.zeros((n_samples, n_params), dtype=float)
    sigma2_chain = np.zeros(n_samples, dtype=float)
    theta_current = theta_init.copy()
    sigma2_current = float(sigma2_init)
    sse_current = sse(theta_current, re_data, pr_data, nu_data)
    log_post_current = -0.5 * n_data * np.log(sigma2_current) - 0.5 * sse_current / sigma2_current + log_prior(theta_current)
    accepted = 0

    for idx in range(n_samples):
        theta_proposal = theta_current + proposal_chol @ rng.standard_normal(n_params)
        if np.all(theta_proposal > 0.0):
            sse_proposal = sse(theta_proposal, re_data, pr_data, nu_data)
            log_post_proposal = (
                -0.5 * n_data * np.log(sigma2_current)
                - 0.5 * sse_proposal / sigma2_current
                + log_prior(theta_proposal)
            )
            if np.log(rng.random()) < log_post_proposal - log_post_current:
                theta_current = theta_proposal
                sse_current = sse_proposal
                log_post_current = log_post_proposal
                accepted += 1

        sigma2_current = invgamma.rvs(
            a=0.5 * n_data,
            scale=0.5 * sse_current,
            random_state=rng,
        )
        log_post_current = -0.5 * n_data * np.log(sigma2_current) - 0.5 * sse_current / sigma2_current + log_prior(theta_current)

        theta_chain[idx] = theta_current
        sigma2_chain[idx] = sigma2_current

    acceptance_rate = accepted / n_samples
    return theta_chain, sigma2_chain, acceptance_rate


def save_residual_plot(predicted: np.ndarray, residual_values: np.ndarray, sigma_hat: float, file_path: Path, title: str) -> None:
    """Save a residual-vs-prediction diagnostic plot."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(predicted, residual_values, "kx", label="Residual")
    ax.axhline(0.0, color="tab:blue", linewidth=2.0)
    ax.axhline(2.0 * sigma_hat, color="tab:red", linewidth=1.8, linestyle="--", label=r"$\pm 2\hat{\sigma}$")
    ax.axhline(-2.0 * sigma_hat, color="tab:red", linewidth=1.8, linestyle="--")
    ax.set_xlabel("Predicted Nu")
    ax.set_ylabel("Residual")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def save_chain_plot(chain: np.ndarray, burn_in: int, file_path: Path) -> None:
    """Save parameter-chain trace plots."""
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    iterations = np.arange(1, len(chain) + 1)
    for idx, ax in enumerate(axes):
        ax.plot(iterations, chain[:, idx], linewidth=0.8)
        ax.axvline(burn_in, color="tab:red", linestyle="--", linewidth=1.2)
        ax.set_ylabel(labels[idx])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("MCMC iteration")
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def save_density_plot(samples: np.ndarray, file_path: Path) -> None:
    """Save posterior density estimates for each parameter."""
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for idx, ax in enumerate(axes):
        mesh = np.linspace(samples[:, idx].min(), samples[:, idx].max(), 300)
        kde = gaussian_kde(samples[:, idx])
        ax.plot(mesh, kde(mesh), color="black", linewidth=2.0)
        ax.set_xlabel(labels[idx])
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def save_pairwise_plot(samples: np.ndarray, file_path: Path) -> None:
    """Save pairwise posterior scatter plots."""
    pairs: Iterable[tuple[int, int]] = ((0, 1), (0, 2), (1, 2))
    labels = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (i, j) in zip(axes, pairs):
        ax.scatter(samples[:, i], samples[:, j], s=6, alpha=0.25)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def save_sigma2_plot(sigma2_chain: np.ndarray, burn_in: int, file_path: Path) -> None:
    """Save the sampled observation-variance chain."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(sigma2_chain) + 1), sigma2_chain, linewidth=0.8)
    ax.axvline(burn_in, color="tab:red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel(r"$\sigma^2$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def save_prediction_interval_plot(
    posterior_theta: np.ndarray,
    posterior_sigma2: np.ndarray,
    re_data: np.ndarray,
    pr_data: np.ndarray,
    nu_data: np.ndarray,
    file_path: Path,
) -> float:
    """Save posterior predictive intervals for the measured operating conditions.

    Returns
    -------
    float
        Empirical coverage of the 95% prediction interval.
    """
    rng = np.random.default_rng(MCMC_SEED + 1)
    n_draws = min(2000, len(posterior_theta))
    draw_indices = np.linspace(0, len(posterior_theta) - 1, n_draws, dtype=int)
    predictive_draws = np.empty((n_draws, len(nu_data)), dtype=float)

    for row, idx in enumerate(draw_indices):
        mean_prediction = dittus_boelter(posterior_theta[idx], re_data, pr_data)
        predictive_draws[row] = mean_prediction + rng.normal(scale=np.sqrt(posterior_sigma2[idx]), size=len(nu_data))

    pred_mean = predictive_draws.mean(axis=0)
    lower = np.percentile(predictive_draws, 2.5, axis=0)
    upper = np.percentile(predictive_draws, 97.5, axis=0)
    coverage = float(np.mean((nu_data >= lower) & (nu_data <= upper)))

    order = np.argsort(pred_mean)
    x_axis = np.arange(1, len(nu_data) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.fill_between(x_axis, lower[order], upper[order], color="lightgray", alpha=0.8, label="95% prediction interval")
    ax.plot(x_axis, pred_mean[order], "k-", linewidth=2.0, label="Posterior predictive mean")
    ax.plot(x_axis, nu_data[order], "ro", ms=4, label="Observed Nu")
    ax.set_xlabel("Sorted experiment index")
    ax.set_ylabel("Nu")
    ax.set_title("Posterior predictive intervals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return coverage


def main() -> None:
    """Run the full Dittus-Boelter calibration and Bayesian analysis workflow."""
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "Figures&Data"
    output_dir.mkdir(exist_ok=True)
    data_path = output_dir / "db_data.txt"

    re_data, pr_data, nu_data = load_db_data(data_path)
    n_data = len(nu_data)
    n_params = len(INITIAL_PARAMS)

    initial_predictions = dittus_boelter(INITIAL_PARAMS, re_data, pr_data)
    initial_residuals = initial_predictions - nu_data
    initial_sigma2 = float(initial_residuals @ initial_residuals / (n_data - n_params))
    initial_sigma_hat = np.sqrt(initial_sigma2)
    save_residual_plot(
        initial_predictions,
        initial_residuals,
        initial_sigma_hat,
        output_dir / "final_initial_residuals.png",
        "Residuals with initial parameters",
    )

    fit_result = estimate_parameters(INITIAL_PARAMS, re_data, pr_data, nu_data)
    fitted_params = fit_result.x
    fitted_predictions = dittus_boelter(fitted_params, re_data, pr_data)
    fitted_residuals = fitted_predictions - nu_data
    sigma2_hat = float(fitted_residuals @ fitted_residuals / (n_data - n_params))
    sigma_hat = np.sqrt(sigma2_hat)

    jacobian = sensitivity_matrix(fitted_params, re_data, pr_data)
    fisher_information = jacobian.T @ jacobian
    covariance = sigma2_hat * np.linalg.inv(fisher_information)
    identifiable, unidentifiable = pss_svd(jacobian, IDENTIFIABILITY_TOL)

    save_residual_plot(
        fitted_predictions,
        fitted_residuals,
        sigma_hat,
        output_dir / "final_fitted_residuals.png",
        "Residuals after nonlinear least squares",
    )

    theta_chain, sigma2_chain, acceptance_rate = run_mcmc(
        theta_init=fitted_params,
        sigma2_init=sigma2_hat,
        proposal_cov=covariance,
        re_data=re_data,
        pr_data=pr_data,
        nu_data=nu_data,
        n_samples=MCMC_SAMPLES,
        seed=MCMC_SEED,
    )

    posterior_theta = theta_chain[MCMC_BURN_IN:]
    posterior_sigma2 = sigma2_chain[MCMC_BURN_IN:]
    posterior_mean = posterior_theta.mean(axis=0)
    posterior_std = posterior_theta.std(axis=0, ddof=1)
    sigma2_posterior_mean = float(posterior_sigma2.mean())

    save_chain_plot(theta_chain, MCMC_BURN_IN, output_dir / "final_parameter_chains.png")
    save_density_plot(posterior_theta, output_dir / "final_parameter_densities.png")
    save_pairwise_plot(posterior_theta, output_dir / "final_pairwise_scatter.png")
    save_sigma2_plot(sigma2_chain, MCMC_BURN_IN, output_dir / "final_sigma2_chain.png")
    prediction_coverage = save_prediction_interval_plot(
        posterior_theta,
        posterior_sigma2,
        re_data,
        pr_data,
        nu_data,
        output_dir / "final_prediction_intervals.png",
    )

    print("=== Project 6 Final: Dittus-Boelter analysis ===")
    print(f"Number of data points: {n_data}")
    print(f"Initial parameters: {INITIAL_PARAMS}")
    print(f"Least-squares parameters: {fitted_params}")
    print(f"Residual standard deviation: {sigma_hat:.6f}")
    print("Fisher information matrix:")
    print(fisher_information)
    print(f"Identifiable parameters (1-based): {identifiable}")
    print(f"Unidentifiable parameters (1-based): {unidentifiable}")
    print(f"MCMC acceptance rate: {acceptance_rate:.3f}")
    print(f"Posterior mean: {posterior_mean}")
    print(f"Posterior std: {posterior_std}")
    print(f"Posterior mean of sigma^2: {sigma2_posterior_mean:.6f}")
    print(f"95% prediction interval coverage: {prediction_coverage:.3f}")
    print(f"Saved outputs in: {output_dir}")


if __name__ == "__main__":
    main()
