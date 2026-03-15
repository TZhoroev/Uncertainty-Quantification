"""
Project 5, Problem 1: Polynomial surrogate models

This script demonstrates:
1. Random sampling vs Latin Hypercube Sampling (LHS)
2. Polynomial regression surrogate model construction
3. Limitations in extrapolation domains
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc


def target_function(q):
    """Target function to approximate."""
    return (6 * q**2 + 3) * np.sin(6 * q - 4)


def latin_hypercube_sample(n, bounds=(0, 2)):
    """Generate Latin Hypercube samples."""
    sampler = qmc.LatinHypercube(d=1)
    samples = sampler.random(n=n)
    samples = qmc.scale(samples, bounds[0], bounds[1])
    return samples.flatten()


def polynomial_fit(q_samples, f_samples, degree):
    """Fit polynomial of given degree to samples."""
    X = np.vander(q_samples, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X, f_samples, rcond=None)[0]
    return coeffs


def evaluate_polynomial(q, coeffs):
    """Evaluate polynomial with given coefficients."""
    X = np.vander(q, len(coeffs), increasing=True)
    return X @ coeffs


def main():
    # Number of samples
    M = 15
    degree = 8
    
    # Generate samples
    np.random.seed(42)
    q_random = 2 * np.random.rand(M)  # Uniform random sampling
    q_lhs = latin_hypercube_sample(M, bounds=(0, 2))
    
    # Evaluate function at sample points
    f_random = target_function(q_random)
    f_lhs = target_function(q_lhs)
    
    # Fit polynomial surrogates
    coeffs_random = polynomial_fit(q_random, f_random, degree)
    coeffs_lhs = polynomial_fit(q_lhs, f_lhs, degree)
    
    # Evaluation grid
    q_plot = np.linspace(0, 2, 200)
    f_true = target_function(q_plot)
    f_surrogate_random = evaluate_polynomial(q_plot, coeffs_random)
    f_surrogate_lhs = evaluate_polynomial(q_plot, coeffs_lhs)
    
    # Extended grid for extrapolation
    q_ext = np.linspace(-0.5, 2.5, 300)
    f_true_ext = target_function(q_ext)
    f_surrogate_ext = evaluate_polynomial(q_ext, coeffs_lhs)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Sampling comparison
    axes[0].plot(q_plot, f_true, '-b', linewidth=2, label='True Response')
    axes[0].plot(q_random, f_random, 'ob', markersize=8, label='Random Samples')
    axes[0].plot(q_lhs, f_lhs, 'dr', markersize=8, label='LHS Samples')
    axes[0].set_xlabel('Parameter q', fontsize=12)
    axes[0].set_ylabel('Model Response', fontsize=12)
    axes[0].set_title('Sampling Methods', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Surrogate fit (calibration domain)
    axes[1].plot(q_plot, f_true, '-b', linewidth=2, label='True Response')
    axes[1].plot(q_plot, f_surrogate_lhs, '--k', linewidth=2, label='Surrogate (LHS)')
    axes[1].plot(q_lhs, f_lhs, 'dr', markersize=8, label='Training Data')
    axes[1].set_xlabel('Parameter q', fontsize=12)
    axes[1].set_ylabel('Model Response', fontsize=12)
    axes[1].set_title('Surrogate Model (Calibration)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Extrapolation behavior
    axes[2].plot(q_ext, f_true_ext, '-b', linewidth=2, label='True Response')
    axes[2].plot(q_ext, f_surrogate_ext, '--k', linewidth=2, label='Surrogate')
    axes[2].plot(q_lhs, f_lhs, 'dr', markersize=8, label='Training Data')
    axes[2].axvline(x=0, color='gray', linestyle=':', alpha=0.7)
    axes[2].axvline(x=2, color='gray', linestyle=':', alpha=0.7)
    axes[2].set_xlabel('Parameter q', fontsize=12)
    axes[2].set_ylabel('Model Response', fontsize=12)
    axes[2].set_title('Surrogate Model (Extrapolation)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures&Data/surrogate_polynomial.png', dpi=150)
    plt.show()
    
    # Compute approximation errors
    rmse_cal = np.sqrt(np.mean((f_true - f_surrogate_lhs)**2))
    rmse_ext = np.sqrt(np.mean((f_true_ext - f_surrogate_ext)**2))
    
    print("=== Surrogate Model Analysis ===")
    print(f"Polynomial degree: {degree}")
    print(f"Number of training samples: {M}")
    print(f"\nRMSE (calibration domain [0,2]): {rmse_cal:.4f}")
    print(f"RMSE (extended domain [-0.5, 2.5]): {rmse_ext:.4f}")
    
    print("\n=== Key Observations ===")
    print("1. LHS provides better coverage of the input space")
    print("2. Polynomial surrogate fits well in calibration domain")
    print("3. Extrapolation outside training domain can be unreliable")


if __name__ == "__main__":
    main()
