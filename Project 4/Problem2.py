"""Bayesian credible and prediction intervals for an aluminum rod heat model.

This script mirrors the MATLAB workflow from Project 4, Problem 2. It

1. loads the aluminum rod temperature data,
2. evaluates the analytical steady-state heat solution,
3. estimates an OLS covariance using complex-step sensitivities,
4. runs a lightweight DRAM-style adaptive MCMC sampler for the posterior of
   heat flux ``Q`` and convection coefficient ``h``,
5. constructs Bayesian credible and posterior-predictive intervals, and
6. plots posterior diagnostics and prediction bands.

Only ``numpy``, ``scipy``, and ``matplotlib`` are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, invgamma

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Figures & Data"

A = 0.95  # cm
B = 0.95  # cm
L = 70.0  # cm
K = 2.37  # W / (cm C)
X_FULL = np.array([10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66], dtype=float)
SAMPLE_SLICE = slice(3, 12)
DEFAULT_Q = -18.41
DEFAULT_H = 0.00191
HEAT_BOUNDS = np.array([[-40.0, -1.0], [1.0e-5, 2.0e-2]], dtype=float)


@dataclass
class HeatMCMCResult:
    """Container for posterior samples and interval summaries."""

    theta_chain: np.ndarray
    sigma2_chain: np.ndarray
    burn_in: int
    accept_rate: float
    delayed_accept_rate: float
    x_full: np.ndarray
    y_full: np.ndarray
    x_obs: np.ndarray
    y_obs: np.ndarray
    posterior_mean: np.ndarray
    sigma2_mean: float
    credible_lower: np.ndarray
    credible_upper: np.ndarray
    predictive_lower: np.ndarray
    predictive_upper: np.ndarray


@dataclass
class FrequentistHeatIntervals:
    """Frequentist interval summary for the heat model."""

    x: np.ndarray
    mean: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    prediction_lower: np.ndarray
    prediction_upper: np.ndarray
    sigma2: float
    covariance: np.ndarray
    data_x: np.ndarray
    data_y: np.ndarray


@dataclass
class SamplerOutput:
    """Raw sampler output."""

    chain: np.ndarray
    sigma2_chain: np.ndarray
    accept_rate: float
    delayed_accept_rate: float


def load_heat_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load full and calibration data for the aluminum rod problem."""

    raw = np.loadtxt(DATA_DIR / "final_al_data.txt")
    y_full = raw[1:16].astype(float)
    u_amb = float(raw[16])
    x_obs = X_FULL[SAMPLE_SLICE]
    y_obs = y_full[SAMPLE_SLICE]
    return X_FULL.copy(), y_full, x_obs, y_obs, u_amb


def heat_solution(x: np.ndarray, q: complex, h: complex, u_amb: float) -> np.ndarray:
    """Evaluate the analytical temperature profile for the rod."""

    x = np.asarray(x)
    gamma = np.sqrt(2.0 * (A + B) * h / (A * B * K))
    f1 = np.exp(gamma * L) * (h + K * gamma)
    f2 = np.exp(-gamma * L) * (h - K * gamma)
    f3 = f1 / (f1 + f2)
    c1 = -(q * f3) / (K * gamma)
    c2 = q / (K * gamma) + c1
    return c1 * np.exp(-gamma * x) + c2 * np.exp(gamma * x) + u_amb


def complex_step_jacobian(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, params: Iterable[float], step: float = 1.0e-20
) -> np.ndarray:
    """Compute a Jacobian with complex-step differentiation."""

    params = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)
    jac = np.empty((x.size, params.size), dtype=float)
    for j in range(params.size):
        perturbed = params.astype(np.complex128)
        perturbed[j] += 1j * step
        jac[:, j] = np.imag(model(x, perturbed)) / step
    return jac


def initial_heat_statistics() -> dict[str, np.ndarray | float]:
    """Construct OLS-like statistics used to initialize the sampler."""

    x_full, y_full, x_obs, y_obs, u_amb = load_heat_data()
    theta0 = np.array([DEFAULT_Q, DEFAULT_H], dtype=float)
    model = lambda x, theta: heat_solution(x, theta[0], theta[1], u_amb)
    y_hat = np.real(model(x_full, theta0))
    residuals = y_full - y_hat
    sigma2 = float(residuals @ residuals / (x_full.size - theta0.size))
    sens = complex_step_jacobian(model, x_full, theta0)
    covariance = sigma2 * np.linalg.inv(sens.T @ sens)
    return {
        "theta0": theta0,
        "sigma2": sigma2,
        "covariance": covariance,
        "x_full": x_full,
        "y_full": y_full,
        "x_obs": x_obs,
        "y_obs": y_obs,
        "u_amb": u_amb,
    }


def _within_bounds(theta: np.ndarray, bounds: np.ndarray) -> bool:
    return bool(np.all(theta > bounds[:, 0]) and np.all(theta < bounds[:, 1]))


def _safe_log_one_minus(alpha: float) -> float:
    return np.log(max(1.0 - min(alpha, 1.0), 1.0e-12))


def dram_mcmc(
    sse_function: Callable[[np.ndarray], float],
    theta0: np.ndarray,
    sigma2_0: float,
    proposal_cov: np.ndarray,
    bounds: np.ndarray,
    n_obs: int,
    nsimu: int = 10_000,
    burn_in: int = 2_000,
    adapt_interval: int = 100,
    seed: int = 7,
) -> SamplerOutput:
    """Run a compact DRAM-style sampler with Gibbs updates for ``sigma^2``."""

    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, dtype=float).copy()
    d = theta.size
    base_cov = np.asarray(proposal_cov, dtype=float).copy()
    jitter = 1.0e-12 * np.eye(d)
    scale1 = 2.38**2 / d
    scale2 = 0.25 * scale1

    chain = np.empty((nsimu, d), dtype=float)
    sigma2_chain = np.empty(nsimu, dtype=float)

    accepted = 0
    delayed_accepted = 0
    current_sse = float(sse_function(theta))
    sigma2 = float(sigma2_0)

    chol1 = np.linalg.cholesky(scale1 * base_cov + jitter)
    chol2 = np.linalg.cholesky(scale2 * base_cov + jitter)
    inv1 = np.linalg.inv(scale1 * base_cov + jitter)

    for i in range(nsimu):
        if i >= max(burn_in, 2 * d) and i % adapt_interval == 0:
            empirical = np.cov(chain[:i].T)
            if empirical.ndim == 0:
                empirical = np.array([[float(empirical)]])
            base_cov = empirical + 1.0e-10 * np.eye(d)
            cov1 = scale1 * base_cov + jitter
            cov2 = scale2 * base_cov + jitter
            chol1 = np.linalg.cholesky(cov1)
            chol2 = np.linalg.cholesky(cov2)
            inv1 = np.linalg.inv(cov1)

        current_log_target = -0.5 * current_sse / sigma2
        first = theta + chol1 @ rng.normal(size=d)
        first_log_target = -np.inf
        alpha1_xy = 0.0

        if _within_bounds(first, bounds):
            first_sse = float(sse_function(first))
            first_log_target = -0.5 * first_sse / sigma2
            alpha1_xy = min(1.0, float(np.exp(min(700.0, first_log_target - current_log_target))))
            if rng.random() < alpha1_xy:
                theta = first
                current_sse = first_sse
                accepted += 1
            else:
                second = theta + chol2 @ rng.normal(size=d)
                if _within_bounds(second, bounds):
                    second_sse = float(sse_function(second))
                    second_log_target = -0.5 * second_sse / sigma2
                    alpha1_zy = 0.0 if not np.isfinite(first_log_target) else min(
                        1.0, float(np.exp(min(700.0, first_log_target - second_log_target)))
                    )
                    quad_num = (first - second) @ inv1 @ (first - second)
                    quad_den = (first - theta) @ inv1 @ (first - theta)
                    log_alpha2 = (
                        second_log_target
                        - current_log_target
                        - 0.5 * (quad_num - quad_den)
                        + _safe_log_one_minus(alpha1_zy)
                        - _safe_log_one_minus(alpha1_xy)
                    )
                    alpha2 = min(1.0, float(np.exp(min(700.0, log_alpha2))))
                    if rng.random() < alpha2:
                        theta = second
                        current_sse = second_sse
                        delayed_accepted += 1
        else:
            second = theta + chol2 @ rng.normal(size=d)
            if _within_bounds(second, bounds):
                second_sse = float(sse_function(second))
                second_log_target = -0.5 * second_sse / sigma2
                quad_num = (first - second) @ inv1 @ (first - second)
                quad_den = (first - theta) @ inv1 @ (first - theta)
                log_alpha2 = second_log_target - current_log_target - 0.5 * (quad_num - quad_den)
                alpha2 = min(1.0, float(np.exp(min(700.0, log_alpha2))))
                if rng.random() < alpha2:
                    theta = second
                    current_sse = second_sse
                    delayed_accepted += 1

        sigma2 = float(invgamma.rvs(a=0.5 * n_obs, scale=0.5 * current_sse, random_state=rng))
        chain[i] = theta
        sigma2_chain[i] = sigma2

    return SamplerOutput(
        chain=chain,
        sigma2_chain=sigma2_chain,
        accept_rate=accepted / nsimu,
        delayed_accept_rate=delayed_accepted / nsimu,
    )


def frequentist_intervals_heat() -> FrequentistHeatIntervals:
    """Construct linearized frequentist confidence and prediction intervals."""

    stats = initial_heat_statistics()
    theta0 = stats["theta0"]
    u_amb = float(stats["u_amb"])
    x_full = np.asarray(stats["x_full"])
    mean = np.real(heat_solution(x_full, theta0[0], theta0[1], u_amb))
    sens = complex_step_jacobian(lambda x, th: heat_solution(x, th[0], th[1], u_amb), x_full, theta0)
    covariance = np.asarray(stats["covariance"])
    sigma2 = float(stats["sigma2"])
    var_model = sens @ covariance @ sens.T
    var_total = var_model + sigma2 * np.eye(x_full.size)
    sd_conf = 2.0 * np.sqrt(np.clip(np.diag(var_model), 0.0, None))
    sd_pred = 2.0 * np.sqrt(np.clip(np.diag(var_total), 0.0, None))
    return FrequentistHeatIntervals(
        x=x_full,
        mean=mean,
        confidence_lower=mean - sd_conf,
        confidence_upper=mean + sd_conf,
        prediction_lower=mean - sd_pred,
        prediction_upper=mean + sd_pred,
        sigma2=sigma2,
        covariance=covariance,
        data_x=x_full,
        data_y=np.asarray(stats["y_full"]),
    )


def run_problem2(
    nsimu: int = 10_000,
    burn_in: int = 2_000,
    posterior_draws: int = 2_000,
    seed: int = 7,
    save_plots: bool = True,
) -> HeatMCMCResult:
    """Execute the full Bayesian workflow for Project 4 Problem 2."""

    stats = initial_heat_statistics()
    theta0 = np.asarray(stats["theta0"])
    sigma2_0 = float(stats["sigma2"])
    proposal_cov = np.asarray(stats["covariance"])
    x_full = np.asarray(stats["x_full"])
    y_full = np.asarray(stats["y_full"])
    x_obs = np.asarray(stats["x_obs"])
    y_obs = np.asarray(stats["y_obs"])
    u_amb = float(stats["u_amb"])

    def sse(theta: np.ndarray) -> float:
        pred = np.real(heat_solution(x_obs, theta[0], theta[1], u_amb))
        residual = y_obs - pred
        return float(residual @ residual)

    sampler = dram_mcmc(
        sse_function=sse,
        theta0=theta0,
        sigma2_0=sigma2_0,
        proposal_cov=proposal_cov,
        bounds=HEAT_BOUNDS,
        n_obs=x_obs.size,
        nsimu=nsimu,
        burn_in=burn_in,
        seed=seed,
    )

    theta_chain = sampler.chain[burn_in:]
    sigma2_chain = sampler.sigma2_chain[burn_in:]
    if theta_chain.shape[0] > posterior_draws:
        draw_idx = np.linspace(0, theta_chain.shape[0] - 1, posterior_draws, dtype=int)
        theta_use = theta_chain[draw_idx]
        sigma2_use = sigma2_chain[draw_idx]
    else:
        theta_use = theta_chain
        sigma2_use = sigma2_chain

    predictions = np.vstack([
        np.real(heat_solution(x_full, theta[0], theta[1], u_amb)) for theta in theta_use
    ])
    rng = np.random.default_rng(seed + 101)
    predictive_samples = predictions + rng.normal(scale=np.sqrt(sigma2_use)[:, None], size=predictions.shape)

    posterior_mean = theta_chain.mean(axis=0)
    sigma2_mean = float(sigma2_chain.mean())
    credible_lower, credible_upper = np.quantile(predictions, [0.025, 0.975], axis=0)
    predictive_lower, predictive_upper = np.quantile(predictive_samples, [0.025, 0.975], axis=0)

    if save_plots:
        save_problem2_figures(
            theta_chain=theta_chain,
            sigma2_chain=sigma2_chain,
            x_full=x_full,
            y_full=y_full,
            x_obs=x_obs,
            y_obs=y_obs,
            predictions=predictions,
            credible_lower=credible_lower,
            credible_upper=credible_upper,
            predictive_lower=predictive_lower,
            predictive_upper=predictive_upper,
        )

    return HeatMCMCResult(
        theta_chain=theta_chain,
        sigma2_chain=sigma2_chain,
        burn_in=burn_in,
        accept_rate=sampler.accept_rate,
        delayed_accept_rate=sampler.delayed_accept_rate,
        x_full=x_full,
        y_full=y_full,
        x_obs=x_obs,
        y_obs=y_obs,
        posterior_mean=posterior_mean,
        sigma2_mean=sigma2_mean,
        credible_lower=credible_lower,
        credible_upper=credible_upper,
        predictive_lower=predictive_lower,
        predictive_upper=predictive_upper,
    )


def save_problem2_figures(
    theta_chain: np.ndarray,
    sigma2_chain: np.ndarray,
    x_full: np.ndarray,
    y_full: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    predictions: np.ndarray,
    credible_lower: np.ndarray,
    credible_upper: np.ndarray,
    predictive_lower: np.ndarray,
    predictive_upper: np.ndarray,
) -> None:
    """Create diagnostic and interval plots for Problem 2."""

    q_vals = theta_chain[:, 0]
    h_vals = theta_chain[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(q_vals, color="black", lw=0.8)
    axes[0, 0].set_title("Posterior chain: Q")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Q")

    axes[0, 1].plot(h_vals, color="black", lw=0.8)
    axes[0, 1].set_title("Posterior chain: h")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("h")

    q_kde = gaussian_kde(q_vals)
    h_kde = gaussian_kde(h_vals)
    q_grid = np.linspace(q_vals.min(), q_vals.max(), 200)
    h_grid = np.linspace(h_vals.min(), h_vals.max(), 200)
    axes[1, 0].plot(q_grid, q_kde(q_grid), color="tab:blue", lw=2)
    axes[1, 0].set_title("Posterior density: Q")
    axes[1, 0].set_xlabel("Q")

    axes[1, 1].plot(h_grid, h_kde(h_grid), color="tab:orange", lw=2)
    axes[1, 1].set_title("Posterior density: h")
    axes[1, 1].set_xlabel("h")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem2_mcmc_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x_full, predictive_lower, predictive_upper, color="0.3", alpha=0.25, label="95% prediction interval")
    ax.fill_between(x_full, credible_lower, credible_upper, color="tab:blue", alpha=0.35, label="95% credible interval")
    ax.plot(x_full, predictions.mean(axis=0), color="black", lw=2, label="Posterior mean response")
    ax.scatter(x_full, y_full, color="crimson", marker="*", s=90, label="Full data")
    ax.scatter(x_obs, y_obs, color="goldenrod", edgecolor="black", s=45, zorder=4, label="Calibration subset")
    ax.set_xlabel("Position x (cm)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Aluminum rod Bayesian intervals")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem2_bayesian_intervals.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the Problem 2 workflow and print a concise summary."""

    result = run_problem2()
    print("Problem 2 complete.")
    print(f"Posterior mean Q = {result.posterior_mean[0]:.4f}")
    print(f"Posterior mean h = {result.posterior_mean[1]:.6f}")
    print(f"Posterior mean sigma^2 = {result.sigma2_mean:.4f}")
    print(f"Stage 1 acceptance = {result.accept_rate:.3f}")
    print(f"Delayed acceptance = {result.delayed_accept_rate:.3f}")
    print(f"Saved plots in: {DATA_DIR}")


if __name__ == "__main__":
    main()
