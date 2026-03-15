"""Bayesian credible and prediction intervals for the SIR infection model.

This script implements the Project 4, Problem 3 workflow using the SIR data in
``Figures & Data/SIR.txt``. It

1. solves the SIR system for the infected population,
2. estimates an OLS covariance using complex-step sensitivities,
3. runs a lightweight DRAM-style adaptive MCMC sampler for ``gamma``, ``delta``,
   and ``r``,
4. propagates posterior uncertainty to the infected population, and
5. plots credible and posterior-predictive intervals together with the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde, invgamma, norm

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Figures & Data"

SIR_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
DEFAULT_THETA = np.array([0.01, 0.1953, 0.7970], dtype=float)
Y0 = np.array([900.0, 100.0, 0.0])
N_POP = 1000.0


@dataclass
class SIRMCMCResult:
    """Posterior results for the SIR calibration."""

    theta_chain: np.ndarray
    sigma2_chain: np.ndarray
    burn_in: int
    accept_rate: float
    delayed_accept_rate: float
    t_data: np.ndarray
    infected_data: np.ndarray
    posterior_mean: np.ndarray
    sigma2_mean: float
    credible_lower: np.ndarray
    credible_upper: np.ndarray
    predictive_lower: np.ndarray
    predictive_upper: np.ndarray


@dataclass
class FrequentistSIRIntervals:
    """Frequentist linearized intervals for the infected trajectory."""

    t: np.ndarray
    mean: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    prediction_lower: np.ndarray
    prediction_upper: np.ndarray
    sigma2: float
    covariance: np.ndarray
    infected_data: np.ndarray


@dataclass
class SamplerOutput:
    """Raw sampler output."""

    chain: np.ndarray
    sigma2_chain: np.ndarray
    accept_rate: float
    delayed_accept_rate: float


def load_sir_data() -> tuple[np.ndarray, np.ndarray]:
    """Load time and infected-population data from disk."""

    data = np.loadtxt(DATA_DIR / "SIR.txt")
    return data[:, 0].astype(float), data[:, 1].astype(float)


def sir_rhs(_t: float, y: np.ndarray, gamma: complex, delta: complex, r: complex, n_pop: float = N_POP) -> np.ndarray:
    """Right-hand side of the SIR model."""

    s, i, recovered = y
    return np.array(
        [
            delta * (n_pop - s) - gamma * i * s,
            gamma * i * s - (r + delta) * i,
            r * i - delta * recovered,
        ],
        dtype=np.result_type(y, gamma, delta, r),
    )


def solve_sir_infected(t_eval: np.ndarray, theta: Iterable[complex], y0: np.ndarray = Y0, n_pop: float = N_POP) -> np.ndarray:
    """Solve the SIR system and return the infected population trajectory."""

    theta = np.asarray(theta)
    gamma, delta, r = theta
    is_complex = np.iscomplexobj(theta)
    dtype = np.complex128 if is_complex else float
    method = "RK45" if is_complex else "LSODA"
    rtol = 1.0e-8 if is_complex else 1.0e-6
    atol = 1.0e-10 if is_complex else 1.0e-8

    y0 = np.asarray(y0, dtype=dtype)
    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        lambda t, y: sir_rhs(t, y, gamma, delta, r, n_pop=n_pop),
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"SIR solve failed: {sol.message}")
    return sol.y[1]


def complex_step_jacobian(
    model: Callable[[np.ndarray, np.ndarray], np.ndarray], x: np.ndarray, params: Iterable[float], step: float = 1.0e-20
) -> np.ndarray:
    """Compute a Jacobian matrix using complex-step differentiation."""

    params = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)
    jac = np.empty((x.size, params.size), dtype=float)
    for j in range(params.size):
        perturbed = params.astype(np.complex128)
        perturbed[j] += 1j * step
        jac[:, j] = np.imag(model(x, perturbed)) / step
    return jac


def initial_sir_statistics() -> dict[str, np.ndarray | float]:
    """Construct the OLS-type covariance used for initialization."""

    t_data, infected_data = load_sir_data()
    theta0 = DEFAULT_THETA.copy()
    infected_model = np.real(solve_sir_infected(t_data, theta0))
    residual = infected_data - infected_model
    sigma2 = float(residual @ residual / (t_data.size - theta0.size))
    sens = complex_step_jacobian(lambda t, th: solve_sir_infected(t, th), t_data, theta0)
    covariance = sigma2 * np.linalg.inv(sens.T @ sens)
    return {
        "theta0": theta0,
        "sigma2": sigma2,
        "covariance": covariance,
        "t_data": t_data,
        "infected_data": infected_data,
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
    seed: int = 13,
) -> SamplerOutput:
    """Run a compact DRAM-style adaptive Metropolis sampler."""

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


def frequentist_intervals_sir() -> FrequentistSIRIntervals:
    """Construct linearized confidence and prediction intervals for ``I(t)``."""

    stats = initial_sir_statistics()
    theta0 = np.asarray(stats["theta0"])
    t_data = np.asarray(stats["t_data"])
    mean = np.real(solve_sir_infected(t_data, theta0))
    sens = complex_step_jacobian(lambda t, th: solve_sir_infected(t, th), t_data, theta0)
    covariance = np.asarray(stats["covariance"])
    sigma2 = float(stats["sigma2"])
    var_model = sens @ covariance @ sens.T
    var_total = var_model + sigma2 * np.eye(t_data.size)
    sd_conf = 2.0 * np.sqrt(np.clip(np.diag(var_model), 0.0, None))
    sd_pred = 2.0 * np.sqrt(np.clip(np.diag(var_total), 0.0, None))
    return FrequentistSIRIntervals(
        t=t_data,
        mean=mean,
        confidence_lower=mean - sd_conf,
        confidence_upper=mean + sd_conf,
        prediction_lower=mean - sd_pred,
        prediction_upper=mean + sd_pred,
        sigma2=sigma2,
        covariance=covariance,
        infected_data=np.asarray(stats["infected_data"]),
    )


def run_problem3(
    nsimu: int = 4_000,
    burn_in: int = 1_000,
    posterior_draws: int = 1_000,
    seed: int = 13,
    save_plots: bool = True,
) -> SIRMCMCResult:
    """Execute the Bayesian SIR uncertainty propagation workflow."""

    stats = initial_sir_statistics()
    theta0 = np.asarray(stats["theta0"])
    sigma2_0 = float(stats["sigma2"])
    proposal_cov = np.asarray(stats["covariance"])
    t_data = np.asarray(stats["t_data"])
    infected_data = np.asarray(stats["infected_data"])

    def sse(theta: np.ndarray) -> float:
        pred = np.real(solve_sir_infected(t_data, theta))
        residual = infected_data - pred
        return float(residual @ residual)

    sampler = dram_mcmc(
        sse_function=sse,
        theta0=theta0,
        sigma2_0=sigma2_0,
        proposal_cov=proposal_cov,
        bounds=SIR_BOUNDS,
        n_obs=t_data.size,
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

    predictions = np.vstack([np.real(solve_sir_infected(t_data, theta)) for theta in theta_use])
    rng = np.random.default_rng(seed + 101)
    predictive_samples = predictions + rng.normal(scale=np.sqrt(sigma2_use)[:, None], size=predictions.shape)

    posterior_mean = theta_chain.mean(axis=0)
    sigma2_mean = float(sigma2_chain.mean())
    credible_lower, credible_upper = np.quantile(predictions, [0.025, 0.975], axis=0)
    predictive_lower, predictive_upper = np.quantile(predictive_samples, [0.025, 0.975], axis=0)

    if save_plots:
        save_problem3_figures(
            theta_chain=theta_chain,
            sigma2_chain=sigma2_chain,
            t_data=t_data,
            infected_data=infected_data,
            predictions=predictions,
            credible_lower=credible_lower,
            credible_upper=credible_upper,
            predictive_lower=predictive_lower,
            predictive_upper=predictive_upper,
        )

    return SIRMCMCResult(
        theta_chain=theta_chain,
        sigma2_chain=sigma2_chain,
        burn_in=burn_in,
        accept_rate=sampler.accept_rate,
        delayed_accept_rate=sampler.delayed_accept_rate,
        t_data=t_data,
        infected_data=infected_data,
        posterior_mean=posterior_mean,
        sigma2_mean=sigma2_mean,
        credible_lower=credible_lower,
        credible_upper=credible_upper,
        predictive_lower=predictive_lower,
        predictive_upper=predictive_upper,
    )


def save_problem3_figures(
    theta_chain: np.ndarray,
    sigma2_chain: np.ndarray,
    t_data: np.ndarray,
    infected_data: np.ndarray,
    predictions: np.ndarray,
    credible_lower: np.ndarray,
    credible_upper: np.ndarray,
    predictive_lower: np.ndarray,
    predictive_upper: np.ndarray,
) -> None:
    """Create diagnostic and interval plots for the SIR posterior."""

    stats = initial_sir_statistics()
    theta0 = np.asarray(stats["theta0"])
    covariance = np.asarray(stats["covariance"])
    labels = [r"$\gamma$", r"$\delta$", "r"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for idx, label in enumerate(labels):
        axes[idx, 0].plot(theta_chain[:, idx], color="black", lw=0.8)
        axes[idx, 0].set_title(f"Posterior chain: {label}")
        axes[idx, 0].set_xlabel("Iteration")
        axes[idx, 0].set_ylabel(label)

        grid = np.linspace(theta_chain[:, idx].min(), theta_chain[:, idx].max(), 200)
        kde = gaussian_kde(theta_chain[:, idx])
        axes[idx, 1].plot(grid, kde(grid), color="tab:blue", lw=2, label="Posterior KDE")
        axes[idx, 1].plot(grid, norm.pdf(grid, theta0[idx], np.sqrt(covariance[idx, idx])), "--", color="crimson", lw=2, label="OLS normal")
        axes[idx, 1].set_title(f"Marginal density: {label}")
        axes[idx, 1].set_xlabel(label)
        axes[idx, 1].legend(fontsize=9)

    for ax in axes.ravel():
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem3_mcmc_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(theta_chain[:, 0], theta_chain[:, 1], s=6, alpha=0.25, color="tab:purple")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel(r"$\delta$")
    axes[0].set_title("Joint posterior: gamma vs delta")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(theta_chain[:, 0], theta_chain[:, 2], s=6, alpha=0.25, color="tab:green")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel("r")
    axes[1].set_title("Joint posterior: gamma vs r")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem3_joint_posteriors.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(t_data, predictive_lower, predictive_upper, color="0.3", alpha=0.25, label="95% prediction interval")
    ax.fill_between(t_data, credible_lower, credible_upper, color="tab:blue", alpha=0.35, label="95% credible interval")
    ax.plot(t_data, predictions.mean(axis=0), color="black", lw=2, label="Posterior mean infected")
    ax.scatter(t_data, infected_data, color="crimson", marker="*", s=80, label="Observed infected data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Infected population")
    ax.set_title("SIR Bayesian uncertainty propagation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem3_sir_intervals.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sigma2_chain, color="black", lw=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\sigma^2$")
    ax.set_title("Measurement-variance chain")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "problem3_sigma2_chain.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Run the Problem 3 workflow and print a concise summary."""

    result = run_problem3()
    gamma, delta, r_value = result.posterior_mean
    print("Problem 3 complete.")
    print(f"Posterior mean gamma = {gamma:.6f}")
    print(f"Posterior mean delta = {delta:.6f}")
    print(f"Posterior mean r = {r_value:.6f}")
    print(f"Posterior mean sigma^2 = {result.sigma2_mean:.4f}")
    print(f"Stage 1 acceptance = {result.accept_rate:.3f}")
    print(f"Delayed acceptance = {result.delayed_accept_rate:.3f}")
    print(f"Saved plots in: {DATA_DIR}")


if __name__ == "__main__":
    main()
