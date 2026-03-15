"""Legendre-polynomial surrogate model for the Project 5 oscillator response.

This script builds a one-dimensional surrogate for

    y(k, t) = 3 cos(sqrt(k) t)

with uncertain stiffness parameter k ~ U[mu_k - sigma_k, mu_k + sigma_k].
The surrogate is represented with Legendre basis functions on the standard
random variable q in [-1, 1], trained with Latin hypercube samples and a
least-squares fit. The script compares surrogate-based response statistics
against direct Monte Carlo estimates and reports accuracy metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import legvander
from scipy.stats import qmc


MU_K = 8.5
SIGMA_K = 1.0e-3
TIME_STEP = 1.0e-3
T_START = 0.0
T_END = 5.0
BASIS_ORDER = 4
N_TRAIN = 15
N_MC = 20000
N_VALIDATION = 4000
RNG_SEED = 42


def response_function(k: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate the oscillator response for arrays of k and t values.

    Parameters
    ----------
    k
        Array of parameter values with shape ``(n_samples,)``.
    t
        Array of time values with shape ``(n_times,)``.

    Returns
    -------
    numpy.ndarray
        Response array with shape ``(n_samples, n_times)``.
    """
    k = np.asarray(k, dtype=float)[:, None]
    t = np.asarray(t, dtype=float)[None, :]
    return 3.0 * np.cos(np.sqrt(k) * t)


def k_to_q(k: np.ndarray, mu_k: float = MU_K, sigma_k: float = SIGMA_K) -> np.ndarray:
    """Map physical stiffness samples to the standard Legendre domain [-1, 1]."""
    return (np.asarray(k, dtype=float) - mu_k) / sigma_k


def q_to_k(q: np.ndarray, mu_k: float = MU_K, sigma_k: float = SIGMA_K) -> np.ndarray:
    """Map standard Legendre-domain samples back to physical stiffness values."""
    return mu_k + sigma_k * np.asarray(q, dtype=float)


def latin_hypercube_samples(n_samples: int, bounds: tuple[float, float], seed: int) -> np.ndarray:
    """Generate one-dimensional Latin hypercube samples in the requested bounds."""
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    samples = sampler.random(n=n_samples)
    scaled = qmc.scale(samples, [bounds[0]], [bounds[1]])
    return scaled.ravel()


def fit_legendre_surrogate(q_train: np.ndarray, y_train: np.ndarray, order: int) -> np.ndarray:
    """Fit Legendre coefficients with least squares.

    Parameters
    ----------
    q_train
        Training locations in the standard domain, shape ``(n_train,)``.
    y_train
        Training responses, shape ``(n_train, n_outputs)``.
    order
        Maximum Legendre polynomial order.

    Returns
    -------
    numpy.ndarray
        Coefficient matrix with shape ``(order + 1, n_outputs)``.
    """
    design_matrix = legvander(q_train, order)
    coefficients, *_ = np.linalg.lstsq(design_matrix, y_train, rcond=None)
    return coefficients


def predict_legendre_surrogate(q: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the Legendre surrogate for one or more q values."""
    design_matrix = legvander(np.asarray(q, dtype=float), coefficients.shape[0] - 1)
    return design_matrix @ coefficients


def batched_statistics(evaluator, samples: np.ndarray, batch_size: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard deviation from batched model evaluations."""
    n_samples = len(samples)
    sample_sum = None
    sample_sum_sq = None

    for start in range(0, n_samples, batch_size):
        batch = samples[start:start + batch_size]
        values = evaluator(batch)
        if sample_sum is None:
            sample_sum = np.zeros(values.shape[1], dtype=float)
            sample_sum_sq = np.zeros(values.shape[1], dtype=float)
        sample_sum += values.sum(axis=0)
        sample_sum_sq += np.square(values).sum(axis=0)

    mean = sample_sum / n_samples
    variance = np.maximum((sample_sum_sq - n_samples * mean**2) / (n_samples - 1), 0.0)
    return mean, np.sqrt(variance)


def main() -> None:
    """Train the Legendre surrogate, assess accuracy, and save diagnostic plots."""
    rng = np.random.default_rng(RNG_SEED)
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "Figures&Data"
    output_dir.mkdir(exist_ok=True)

    t = np.arange(T_START, T_END + TIME_STEP, TIME_STEP)
    k_bounds = (MU_K - SIGMA_K, MU_K + SIGMA_K)

    k_train = np.sort(latin_hypercube_samples(N_TRAIN, k_bounds, seed=RNG_SEED))
    q_train = k_to_q(k_train)
    y_train = response_function(k_train, t)
    coefficients = fit_legendre_surrogate(q_train, y_train, BASIS_ORDER - 1)

    k_mc = rng.uniform(*k_bounds, size=N_MC)
    q_mc = k_to_q(k_mc)
    mc_mean, mc_std = batched_statistics(lambda batch: response_function(batch, t), k_mc)
    surrogate_mean, surrogate_std = batched_statistics(
        lambda batch: predict_legendre_surrogate(batch, coefficients),
        q_mc,
    )

    k_validation = rng.uniform(*k_bounds, size=N_VALIDATION)
    q_validation = k_to_q(k_validation)
    y_true_validation = response_function(k_validation, t)
    y_sur_validation = predict_legendre_surrogate(q_validation, coefficients)
    response_rmse = np.sqrt(np.mean((y_true_validation - y_sur_validation) ** 2))
    response_max_error = np.max(np.abs(y_true_validation - y_sur_validation))
    mean_rmse = np.sqrt(np.mean((mc_mean - surrogate_mean) ** 2))
    std_rmse = np.sqrt(np.mean((mc_std - surrogate_std) ** 2))

    k_plot = np.linspace(*k_bounds, 250)
    q_plot = k_to_q(k_plot)
    snapshot_times = np.array([0.5, 2.5, 5.0])
    snapshot_indices = np.searchsorted(t, snapshot_times)
    true_snapshots = response_function(k_plot, snapshot_times)
    surrogate_snapshots = predict_legendre_surrogate(q_plot, coefficients[:, snapshot_indices])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].plot(t, mc_mean, "b-", linewidth=2.5, label="Monte Carlo mean")
    axes[0].plot(t, surrogate_mean, "k--", linewidth=2.0, label="Legendre surrogate mean")
    axes[0].set_xlabel("Time t")
    axes[0].set_ylabel("Response mean")
    axes[0].set_title("Mean comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, mc_std, "b-", linewidth=2.5, label="Monte Carlo std")
    axes[1].plot(t, surrogate_std, "k--", linewidth=2.0, label="Legendre surrogate std")
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Response standard deviation")
    axes[1].set_title("Standard deviation comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for time_value, y_true, y_sur in zip(snapshot_times, true_snapshots.T, surrogate_snapshots.T):
        axes[2].plot(k_plot, y_true, linewidth=2.0, label=fr"True, t={time_value:g}")
        axes[2].plot(k_plot, y_sur, "--", linewidth=1.8, label=fr"Surrogate, t={time_value:g}")
    axes[2].plot(k_train, response_function(k_train, np.array([snapshot_times[1]])).ravel(), "ko", ms=4, label="Training samples")
    axes[2].set_xlabel("Parameter k")
    axes[2].set_ylabel("Response")
    axes[2].set_title("Surrogate snapshots")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(output_dir / "problem2_legendre_surrogate.png", dpi=150)
    plt.close(fig)

    print("=== Project 5 Problem 2: Legendre surrogate ===")
    print(f"Legendre basis order: {BASIS_ORDER - 1}")
    print(f"Training points (LHS): {N_TRAIN}")
    print(f"Monte Carlo samples: {N_MC}")
    print(f"Validation samples: {N_VALIDATION}")
    print(f"Response RMSE on validation set: {response_rmse:.6e}")
    print(f"Maximum absolute validation error: {response_max_error:.6e}")
    print(f"Mean-curve RMSE: {mean_rmse:.6e}")
    print(f"Std-curve RMSE: {std_rmse:.6e}")
    print(f"Saved figure: {output_dir / 'problem2_legendre_surrogate.png'}")


if __name__ == "__main__":
    main()
