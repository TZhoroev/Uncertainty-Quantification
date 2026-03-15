"""Problem 2: Metropolis-Hastings and DRAM for the SIR model.

This script reproduces the Project 3 uncertainty-quantification workflow in
Python using NumPy, SciPy, and Matplotlib. It

1. loads the SIR data set,
2. computes an ordinary least-squares (OLS) fit,
3. evaluates complex-step sensitivities and the Fisher-information covariance,
4. runs both a standard random-walk Metropolis-Hastings sampler and a
   delayed-rejection adaptive Metropolis (DRAM) sampler, and
5. saves trace, density, and pairwise-scatter plots in ``Figures&Data``.

Examples
--------
Run with defaults::

    python Problem2.py

Use fewer samples for a quick smoke test::

    python Problem2.py --nsimu 1500 --burn 300
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import gaussian_kde, norm


@dataclass
class SamplerResult:
    """Container for MCMC outputs."""

    chain: np.ndarray
    sigma2_chain: np.ndarray
    accept_rate: float
    stage1_accept_rate: float
    stage2_accept_rate: float
    label: str


def sir_rhs(y: np.ndarray, theta: np.ndarray, population: float) -> np.ndarray:
    """Return the SIR right-hand side for the parameter vector ``theta``."""
    gamma, delta, r_param = theta
    s_val, i_val, r_val = y
    return np.array(
        [
            delta * (population - s_val) - gamma * i_val * s_val,
            gamma * i_val * s_val - (r_param + delta) * i_val,
            r_param * i_val - delta * r_val,
        ],
        dtype=np.result_type(y, theta),
    )


def rk4_sir(
    t_eval: np.ndarray,
    theta: np.ndarray,
    y0: np.ndarray,
    population: float,
    n_substeps: int = 10,
) -> np.ndarray:
    """Integrate the SIR model with a fixed-step RK4 scheme.

    A fixed-step solver is used so that complex-step perturbations propagate
    reliably when sensitivities are computed.
    """
    t_eval = np.asarray(t_eval, dtype=float)
    dtype = np.result_type(theta, y0)
    y = np.asarray(y0, dtype=dtype)
    states = np.empty((t_eval.size, 3), dtype=dtype)
    states[0] = y

    for idx in range(1, t_eval.size):
        dt_total = float(t_eval[idx] - t_eval[idx - 1])
        step_count = max(1, int(n_substeps))
        h = dt_total / step_count
        for _ in range(step_count):
            k1 = sir_rhs(y, theta, population)
            k2 = sir_rhs(y + 0.5 * h * k1, theta, population)
            k3 = sir_rhs(y + 0.5 * h * k2, theta, population)
            k4 = sir_rhs(y + h * k3, theta, population)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[idx] = y
    return states


def infected_counts(
    theta: np.ndarray,
    t_eval: np.ndarray,
    infected_data: np.ndarray,
    y0: np.ndarray,
    population: float,
    n_substeps: int,
) -> np.ndarray:
    """Return model-predicted infected counts."""
    del infected_data
    return rk4_sir(t_eval, theta, y0, population, n_substeps=n_substeps)[:, 1]


def residuals(
    theta: np.ndarray,
    t_eval: np.ndarray,
    infected_data: np.ndarray,
    y0: np.ndarray,
    population: float,
    n_substeps: int,
) -> np.ndarray:
    """Residual vector for least-squares fitting."""
    return infected_counts(theta, t_eval, infected_data, y0, population, n_substeps) - infected_data


def sum_of_squares(
    theta: np.ndarray,
    t_eval: np.ndarray,
    infected_data: np.ndarray,
    y0: np.ndarray,
    population: float,
    n_substeps: int,
    bounds: tuple[np.ndarray, np.ndarray],
) -> float:
    """Return the residual sum of squares, enforcing simple box constraints."""
    lower, upper = bounds
    theta = np.asarray(theta, dtype=float)
    if np.any(theta <= lower) or np.any(theta >= upper):
        return np.inf
    res = residuals(theta, t_eval, infected_data, y0, population, n_substeps)
    return float(res @ res)


def complex_step_jacobian(
    theta: np.ndarray,
    t_eval: np.ndarray,
    y0: np.ndarray,
    population: float,
    n_substeps: int,
    step: float = 1e-30,
) -> np.ndarray:
    """Compute sensitivities of infected counts using the complex-step method."""
    theta = np.asarray(theta, dtype=float)
    n_obs = t_eval.size
    n_params = theta.size
    jac = np.empty((n_obs, n_params), dtype=float)

    for j in range(n_params):
        theta_cs = theta.astype(np.complex128)
        theta_cs[j] += 1j * step
        states_cs = rk4_sir(t_eval, theta_cs, y0.astype(np.complex128), population, n_substeps=n_substeps)
        jac[:, j] = np.imag(states_cs[:, 1]) / step
    return jac


def safe_cholesky(cov: np.ndarray) -> np.ndarray:
    """Return a numerically safe Cholesky factor."""
    cov = np.asarray(cov, dtype=float)
    eye = np.eye(cov.shape[0])
    jitter = 1e-12
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    return np.linalg.cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T)


def sample_sigma2(rng: np.random.Generator, sse: float, n_obs: int, sigma2_ref: float, n0: float = 1.0) -> float:
    """Sample the error variance from its inverse-gamma full conditional."""
    shape = 0.5 * (n0 + n_obs)
    scale_term = 0.5 * (n0 * sigma2_ref + sse)
    gamma_draw = rng.gamma(shape=shape, scale=1.0 / scale_term)
    return 1.0 / gamma_draw


def acceptance_probability(log_ratio: float) -> float:
    """Map a log acceptance ratio to ``[0, 1]`` safely."""
    if not np.isfinite(log_ratio):
        return 0.0
    return float(min(1.0, np.exp(min(0.0, log_ratio)) if log_ratio < 0.0 else 1.0))


def run_mcmc(
    label: str,
    theta0: np.ndarray,
    sigma2_0: float,
    proposal_cov: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    sse_function,
    n_obs: int,
    adapt: bool,
    burn: int,
    adapt_interval: int,
    dr_scale: float,
    sigma2_ref: float,
) -> SamplerResult:
    """Run either standard MH or DRAM."""
    theta = np.asarray(theta0, dtype=float).copy()
    dim = theta.size
    chain = np.empty((n_samples, dim), dtype=float)
    sigma2_chain = np.empty(n_samples, dtype=float)
    current_sse = float(sse_function(theta))
    current_sigma2 = float(sigma2_0)

    scaling = (2.38 ** 2) / dim
    proposal_cov_current = scaling * np.asarray(proposal_cov, dtype=float)
    delayed_cov = (dr_scale ** 2) * proposal_cov_current
    accept_stage1 = 0
    accept_stage2 = 0

    for i in range(n_samples):
        chol1 = safe_cholesky(proposal_cov_current)
        cand1 = theta + chol1 @ rng.normal(size=dim)
        sse1 = float(sse_function(cand1))
        log_alpha1 = -0.5 * (sse1 - current_sse) / current_sigma2
        alpha1 = acceptance_probability(log_alpha1)

        accepted = False
        if rng.random() < alpha1:
            theta = cand1
            current_sse = sse1
            accept_stage1 += 1
            accepted = True
        elif adapt:
            chol2 = safe_cholesky(delayed_cov)
            cand2 = theta + chol2 @ rng.normal(size=dim)
            sse2 = float(sse_function(cand2))
            log_alpha1_reverse = -0.5 * (sse1 - sse2) / current_sigma2
            alpha1_reverse = acceptance_probability(log_alpha1_reverse)
            numerator = np.exp(np.clip(-0.5 * (sse2 - current_sse) / current_sigma2, -700.0, 700.0))
            ratio = numerator * (1.0 - alpha1_reverse) / max(1e-12, 1.0 - alpha1)
            alpha2 = float(min(1.0, max(0.0, ratio))) if np.isfinite(sse2) else 0.0
            if rng.random() < alpha2:
                theta = cand2
                current_sse = sse2
                accept_stage2 += 1
                accepted = True

        if not accepted and not np.isfinite(current_sse):
            raise RuntimeError(f'{label} sampler lost a finite state.')

        current_sigma2 = sample_sigma2(rng, current_sse, n_obs=n_obs, sigma2_ref=sigma2_ref)
        chain[i] = theta
        sigma2_chain[i] = current_sigma2

        if adapt and (i + 1) >= max(burn, dim + 5) and (i + 1) % adapt_interval == 0:
            empirical_cov = np.cov(chain[: i + 1].T)
            proposal_cov_current = scaling * (empirical_cov + 1e-10 * np.eye(dim))
            delayed_cov = (dr_scale ** 2) * proposal_cov_current

    total_accepts = accept_stage1 + accept_stage2
    return SamplerResult(
        chain=chain,
        sigma2_chain=sigma2_chain,
        accept_rate=total_accepts / n_samples,
        stage1_accept_rate=accept_stage1 / n_samples,
        stage2_accept_rate=accept_stage2 / n_samples,
        label=label,
    )


def kde_xy(samples: np.ndarray, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth density estimate for one-dimensional samples."""
    samples = np.asarray(samples, dtype=float)
    if grid is None:
        low, high = np.quantile(samples, [0.005, 0.995])
        pad = 0.15 * max(high - low, np.std(samples), 1e-10)
        grid = np.linspace(low - pad, high + pad, 300)
    kde = gaussian_kde(samples)
    return grid, kde(grid)


def scatter_matrix(samples: np.ndarray, names: list[str], title: str, out_path: Path) -> None:
    """Save a lower-triangular pairwise scatter-plot matrix."""
    dim = samples.shape[1]
    subset = samples
    if samples.shape[0] > 3000:
        idx = np.linspace(0, samples.shape[0] - 1, 3000, dtype=int)
        subset = samples[idx]

    fig, axes = plt.subplots(dim, dim, figsize=(3.2 * dim, 3.2 * dim))
    for i in range(dim):
        for j in range(dim):
            ax = axes[i, j]
            if i == j:
                grid, density = kde_xy(subset[:, i])
                ax.plot(grid, density, color='black', lw=2)
            elif i > j:
                ax.scatter(subset[:, j], subset[:, i], s=6, alpha=0.35, color='tab:blue', edgecolors='none')
            else:
                ax.axis('off')
                continue
            if i == dim - 1:
                ax.set_xlabel(names[j])
            if j == 0 and i > 0:
                ax.set_ylabel(names[i])
            ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nsimu', type=int, default=10_000, help='Number of MCMC samples per sampler.')
    parser.add_argument('--burn', type=int, default=2_000, help='Burn-in samples discarded in diagnostics.')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed.')
    parser.add_argument('--adapt-interval', type=int, default=50, help='Covariance adaptation interval for DRAM.')
    parser.add_argument('--n-substeps', type=int, default=10, help='RK4 substeps per observation interval.')
    return parser


def main() -> None:
    """Execute the full Problem 2 workflow."""
    args = build_argument_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'Figures&Data'
    data = np.loadtxt(data_dir / 'SIR.txt')
    t_data = data[:, 0]
    infected_data = data[:, 1]

    y0 = np.array([900.0, 100.0, 0.0])
    population = 1000.0
    theta_init = np.array([0.01, 0.1953, 0.7970])
    bounds = (np.array([1e-6, 1e-6, 1e-6]), np.array([0.25, 1.0, 2.5]))
    param_names = ['gamma', 'delta', 'r']

    fit = least_squares(
        residuals,
        theta_init,
        bounds=bounds,
        args=(t_data, infected_data, y0, population, args.n_substeps),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    theta_hat = fit.x
    res_hat = residuals(theta_hat, t_data, infected_data, y0, population, args.n_substeps)
    sse_hat = float(res_hat @ res_hat)
    n_obs = t_data.size
    n_params = theta_hat.size
    sigma2_hat = sse_hat / max(1, n_obs - n_params)

    jac = complex_step_jacobian(theta_hat, t_data, y0, population, args.n_substeps)
    fisher = jac.T @ jac
    covariance = sigma2_hat * np.linalg.pinv(fisher)
    covariance = 0.5 * (covariance + covariance.T)

    def sse_fn(theta: np.ndarray) -> float:
        return sum_of_squares(theta, t_data, infected_data, y0, population, args.n_substeps, bounds)

    rng_mh = np.random.default_rng(args.seed)
    rng_dram = np.random.default_rng(args.seed + 1)
    mh = run_mcmc(
        label='Metropolis-Hastings',
        theta0=theta_hat,
        sigma2_0=sigma2_hat,
        proposal_cov=covariance,
        n_samples=args.nsimu,
        rng=rng_mh,
        sse_function=sse_fn,
        n_obs=n_obs,
        adapt=False,
        burn=args.burn,
        adapt_interval=args.adapt_interval,
        dr_scale=0.35,
        sigma2_ref=sigma2_hat,
    )
    dram = run_mcmc(
        label='DRAM',
        theta0=theta_hat,
        sigma2_0=sigma2_hat,
        proposal_cov=covariance,
        n_samples=args.nsimu,
        rng=rng_dram,
        sse_function=sse_fn,
        n_obs=n_obs,
        adapt=True,
        burn=args.burn,
        adapt_interval=args.adapt_interval,
        dr_scale=0.35,
        sigma2_ref=sigma2_hat,
    )

    burn = min(max(args.burn, 0), args.nsimu - 1)
    mh_post = mh.chain[burn:]
    dram_post = dram.chain[burn:]
    mh_sigma_post = mh.sigma2_chain[burn:]
    dram_sigma_post = dram.sigma2_chain[burn:]

    trace_path = data_dir / 'problem2_chain_traces.png'
    fig, axes = plt.subplots(n_params + 1, 2, figsize=(13, 3.0 * (n_params + 1)), sharex='col')
    for j, name in enumerate(param_names):
        axes[j, 0].plot(mh.chain[:, j], lw=0.7, color='tab:blue')
        axes[j, 0].set_ylabel(name)
        axes[j, 0].set_title('Metropolis-Hastings')
        axes[j, 1].plot(dram.chain[:, j], lw=0.7, color='tab:orange')
        axes[j, 1].set_ylabel(name)
        axes[j, 1].set_title('DRAM')
    axes[-1, 0].plot(mh.sigma2_chain, lw=0.7, color='tab:blue')
    axes[-1, 0].set_ylabel('sigma^2')
    axes[-1, 1].plot(dram.sigma2_chain, lw=0.7, color='tab:orange')
    axes[-1, 1].set_ylabel('sigma^2')
    axes[-1, 0].set_xlabel('Iteration')
    axes[-1, 1].set_xlabel('Iteration')
    fig.tight_layout()
    fig.savefig(trace_path, dpi=200)
    plt.close(fig)

    density_path = data_dir / 'problem2_parameter_densities.png'
    fig, axes = plt.subplots(n_params, 1, figsize=(9, 3.2 * n_params))
    if n_params == 1:
        axes = np.array([axes])
    for j, name in enumerate(param_names):
        std_j = float(np.sqrt(max(covariance[j, j], 1e-16)))
        x_min = min(mh_post[:, j].min(), dram_post[:, j].min(), theta_hat[j] - 4.0 * std_j)
        x_max = max(mh_post[:, j].max(), dram_post[:, j].max(), theta_hat[j] + 4.0 * std_j)
        grid = np.linspace(x_min, x_max, 350)
        _, mh_density = kde_xy(mh_post[:, j], grid)
        _, dram_density = kde_xy(dram_post[:, j], grid)
        axes[j].plot(grid, mh_density, label='Metropolis-Hastings', color='tab:blue', lw=2)
        axes[j].plot(grid, dram_density, label='DRAM', color='tab:orange', lw=2)
        axes[j].plot(grid, norm.pdf(grid, loc=theta_hat[j], scale=std_j), '--', color='black', lw=2, label='OLS normal')
        axes[j].set_xlabel(name)
        axes[j].set_ylabel('Density')
        axes[j].legend()
    fig.tight_layout()
    fig.savefig(density_path, dpi=200)
    plt.close(fig)

    sigma_density_path = data_dir / 'problem2_sigma2_density.png'
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sigma_grid = np.linspace(min(mh_sigma_post.min(), dram_sigma_post.min()), max(mh_sigma_post.max(), dram_sigma_post.max()), 350)
    _, mh_sigma_density = kde_xy(mh_sigma_post, sigma_grid)
    _, dram_sigma_density = kde_xy(dram_sigma_post, sigma_grid)
    ax.plot(sigma_grid, mh_sigma_density, lw=2, label='Metropolis-Hastings', color='tab:blue')
    ax.plot(sigma_grid, dram_sigma_density, lw=2, label='DRAM', color='tab:orange')
    ax.set_xlabel('sigma^2')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(sigma_density_path, dpi=200)
    plt.close(fig)

    scatter_matrix(mh_post, param_names, 'Problem 2 pairwise scatter: Metropolis-Hastings', data_dir / 'problem2_pairs_mh.png')
    scatter_matrix(dram_post, param_names, 'Problem 2 pairwise scatter: DRAM', data_dir / 'problem2_pairs_dram.png')

    fitted_path = data_dir / 'problem2_sir_fit.png'
    fitted_infected = infected_counts(theta_hat, t_data, infected_data, y0, population, args.n_substeps)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(t_data, infected_data, 'o', label='Data', color='black')
    ax.plot(t_data, fitted_infected, '-', label='OLS fit', color='tab:red', lw=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Infected count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fitted_path, dpi=200)
    plt.close(fig)

    print('Problem 2 complete.')
    print(f'OLS estimate: gamma={theta_hat[0]:.6f}, delta={theta_hat[1]:.6f}, r={theta_hat[2]:.6f}')
    print(f'sigma^2 estimate: {sigma2_hat:.6f}')
    print(f'MH acceptance: {mh.accept_rate:.3f}')
    print(f'DRAM acceptance: {dram.accept_rate:.3f} (stage 1: {dram.stage1_accept_rate:.3f}, stage 2: {dram.stage2_accept_rate:.3f})')
    print('Saved figures:')
    for path in [trace_path, density_path, sigma_density_path, data_dir / 'problem2_pairs_mh.png', data_dir / 'problem2_pairs_dram.png', fitted_path]:
        print(f'  - {path.name}')


if __name__ == '__main__':
    main()
