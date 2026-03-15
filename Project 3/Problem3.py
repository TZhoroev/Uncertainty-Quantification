"""Problem 3: Metropolis-Hastings and DRAM for the Helmholtz model.

This script implements the Project 3 Helmholtz-parameter inference problem in
Python using NumPy, SciPy, and Matplotlib. It

1. loads the Helmholtz data,
2. computes the least-squares estimate and covariance,
3. runs a standard Metropolis-Hastings chain and a DRAM chain, and
4. saves trace, density, and pairwise-scatter plots in ``Figures&Data``.

Examples
--------
Run with defaults::

    python Problem3.py

Use fewer samples for a smoke test::

    python Problem3.py --nsimu 1500 --burn 300
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


@dataclass
class SamplerResult:
    """Container for MCMC outputs."""

    chain: np.ndarray
    sigma2_chain: np.ndarray
    accept_rate: float
    stage1_accept_rate: float
    stage2_accept_rate: float
    label: str


def helmholtz_prediction(params: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Evaluate the polynomial Helmholtz model."""
    alpha1, alpha11, alpha111 = params
    return alpha1 * pressure**2 + alpha11 * pressure**4 + alpha111 * pressure**6


def sum_of_squares(params: np.ndarray, pressure: np.ndarray, psi_data: np.ndarray) -> float:
    """Residual sum of squares for the Helmholtz model."""
    pred = helmholtz_prediction(np.asarray(params, dtype=float), pressure)
    res = pred - psi_data
    return float(res @ res)


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
            alpha2 = float(min(1.0, max(0.0, ratio)))
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
    parser.add_argument('--seed', type=int, default=54321, help='Random seed.')
    parser.add_argument('--adapt-interval', type=int, default=50, help='Covariance adaptation interval for DRAM.')
    return parser


def main() -> None:
    """Execute the full Problem 3 workflow."""
    args = build_argument_parser().parse_args()
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / 'Figures&Data'
    data = np.loadtxt(data_dir / 'Helmholtz.txt')
    pressure = data[:, 0]
    psi_data = data[:, 1]

    design = np.column_stack((pressure**2, pressure**4, pressure**6))
    theta_hat, *_ = np.linalg.lstsq(design, psi_data, rcond=None)
    residual = design @ theta_hat - psi_data
    sse_hat = float(residual @ residual)
    n_obs = pressure.size
    n_params = theta_hat.size
    sigma2_hat = sse_hat / max(1, n_obs - n_params)
    covariance = sigma2_hat * np.linalg.pinv(design.T @ design)
    covariance = 0.5 * (covariance + covariance.T)
    param_names = ['alpha1', 'alpha11', 'alpha111']

    def sse_fn(theta: np.ndarray) -> float:
        return sum_of_squares(theta, pressure, psi_data)

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

    trace_path = data_dir / 'problem3_chain_traces.png'
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

    density_path = data_dir / 'problem3_parameter_densities.png'
    fig, axes = plt.subplots(n_params + 1, 1, figsize=(9, 3.2 * (n_params + 1)))
    for j, name in enumerate(param_names):
        x_min = min(mh_post[:, j].min(), dram_post[:, j].min())
        x_max = max(mh_post[:, j].max(), dram_post[:, j].max())
        grid = np.linspace(x_min, x_max, 350)
        _, mh_density = kde_xy(mh_post[:, j], grid)
        _, dram_density = kde_xy(dram_post[:, j], grid)
        axes[j].plot(grid, mh_density, label='Metropolis-Hastings', color='tab:blue', lw=2)
        axes[j].plot(grid, dram_density, label='DRAM', color='tab:orange', lw=2)
        axes[j].set_xlabel(name)
        axes[j].set_ylabel('Density')
        axes[j].legend()
    sigma_grid = np.linspace(min(mh_sigma_post.min(), dram_sigma_post.min()), max(mh_sigma_post.max(), dram_sigma_post.max()), 350)
    _, mh_sigma_density = kde_xy(mh_sigma_post, sigma_grid)
    _, dram_sigma_density = kde_xy(dram_sigma_post, sigma_grid)
    axes[-1].plot(sigma_grid, mh_sigma_density, lw=2, label='Metropolis-Hastings', color='tab:blue')
    axes[-1].plot(sigma_grid, dram_sigma_density, lw=2, label='DRAM', color='tab:orange')
    axes[-1].set_xlabel('sigma^2')
    axes[-1].set_ylabel('Density')
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(density_path, dpi=200)
    plt.close(fig)

    scatter_matrix(mh_post, param_names, 'Problem 3 pairwise scatter: Metropolis-Hastings', data_dir / 'problem3_pairs_mh.png')
    scatter_matrix(dram_post, param_names, 'Problem 3 pairwise scatter: DRAM', data_dir / 'problem3_pairs_dram.png')

    fit_path = data_dir / 'problem3_helmholtz_fit.png'
    psi_fit = helmholtz_prediction(theta_hat, pressure)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(pressure, psi_data, 'o', label='Data', color='black')
    ax.plot(pressure, psi_fit, '-', label='Least-squares fit', color='tab:red', lw=2)
    ax.set_xlabel('Pressure')
    ax.set_ylabel('psi')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fit_path, dpi=200)
    plt.close(fig)

    print('Problem 3 complete.')
    print(f'Least-squares estimate: alpha1={theta_hat[0]:.6f}, alpha11={theta_hat[1]:.6f}, alpha111={theta_hat[2]:.6f}')
    print(f'sigma^2 estimate: {sigma2_hat:.6f}')
    print(f'MH acceptance: {mh.accept_rate:.3f}')
    print(f'DRAM acceptance: {dram.accept_rate:.3f} (stage 1: {dram.stage1_accept_rate:.3f}, stage 2: {dram.stage2_accept_rate:.3f})')
    print('Saved figures:')
    for path in [trace_path, density_path, data_dir / 'problem3_pairs_mh.png', data_dir / 'problem3_pairs_dram.png', fit_path]:
        print(f'  - {path.name}')


if __name__ == '__main__':
    main()
