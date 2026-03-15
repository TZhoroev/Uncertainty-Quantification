"""
UQ Problem 9.6: Global Sensitivity Analysis for Helmholtz Energy Model

This script demonstrates:
1. Morris screening method
2. Sobol indices (first-order and total-order)
3. Saltelli algorithm for efficient Sobol index computation

Helmholtz energy model: psi(P) = alpha_1*P^2 + alpha_11*P^4 + alpha_111*P^6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def helmholtz_energy(P, alpha_1, alpha_11, alpha_111):
    """Helmholtz energy function."""
    return alpha_1 * P**2 + alpha_11 * P**4 + alpha_111 * P**6


def integrated_helmholtz(alpha_1, alpha_11, alpha_111, P_max=0.8):
    """Integrated Helmholtz energy from 0 to P_max."""
    return (alpha_1 * P_max**3) / 3 + (alpha_11 * P_max**5) / 5 + (alpha_111 * P_max**7) / 7


def morris_screening(func, bounds, r=50, delta=1/20):
    """
    Morris screening method for global sensitivity analysis.
    
    Parameters:
    -----------
    func : callable
        Function to analyze
    bounds : list of tuples
        Parameter bounds [(a1, b1), (a2, b2), ...]
    r : int
        Number of trajectories
    delta : float
        Step size for elementary effects
    
    Returns:
    --------
    mu : array
        Mean of elementary effects
    mu_star : array
        Mean of absolute elementary effects
    sigma : array
        Standard deviation of elementary effects
    """
    p = len(bounds)
    d = np.zeros((p, r))
    
    for i in range(r):
        # Sample base point uniformly
        base = np.array([np.random.uniform(a, b) for a, b in bounds])
        
        # Compute elementary effects for each parameter
        for j in range(p):
            perturbed = base.copy()
            perturbed[j] += delta
            d[j, i] = (func(*perturbed) - func(*base)) / delta
    
    mu = np.mean(d, axis=1)
    mu_star = np.mean(np.abs(d), axis=1)
    sigma = np.std(d, axis=1, ddof=1)
    
    return mu, mu_star, sigma


def sobol_indices(func, bounds, M=10000):
    """
    Compute Sobol sensitivity indices using Saltelli's algorithm.
    
    Parameters:
    -----------
    func : callable
        Function to analyze
    bounds : list of tuples
        Parameter bounds [(a1, b1), (a2, b2), ...]
    M : int
        Number of Monte Carlo samples
    
    Returns:
    --------
    S1 : array
        First-order Sobol indices
    ST : array
        Total-order Sobol indices
    """
    p = len(bounds)
    
    # Generate Sobol sequence (using quasi-random sampling)
    # For simplicity, we use uniform random here
    np.random.seed(42)
    D = np.random.rand(2 * M, p)
    
    # Scale to parameter bounds
    for j, (a, b) in enumerate(bounds):
        D[:, j] = D[:, j] * (b - a) + a
    
    A = D[:M, :]
    B = D[M:2*M, :]
    
    # Evaluate model at A and B
    f_A = np.array([func(*a) for a in A])
    f_B = np.array([func(*b) for b in B])
    
    # Combined outputs for variance estimation
    f_D = np.concatenate([f_A, f_B])
    var_total = np.var(f_D)
    
    S1 = np.zeros(p)
    ST = np.zeros(p)
    
    for j in range(p):
        # Create C_j matrix (A with j-th column from B)
        C_j = A.copy()
        C_j[:, j] = B[:, j]
        f_C = np.array([func(*c) for c in C_j])
        
        # First-order index
        S1[j] = (np.mean(f_B * f_C) - np.mean(f_B) * np.mean(f_A)) / var_total
        
        # Total-order index
        ST[j] = 0.5 * np.mean((f_A - f_C)**2) / var_total
    
    return S1, ST


def main():
    # Helmholtz energy model parameters
    alpha_1 = -389.4
    alpha_11 = 761.3
    alpha_111 = 61.5
    
    # (a) Plot Helmholtz energy
    P_vals = np.linspace(-0.8, 0.8, 161)
    psi_vals = helmholtz_energy(P_vals, alpha_1, alpha_11, alpha_111)
    
    plt.figure(figsize=(10, 6))
    plt.plot(P_vals, psi_vals, '-k', linewidth=2)
    plt.xlabel('Polarization P', fontsize=14)
    plt.ylabel('Helmholtz Energy', fontsize=14)
    plt.title('Helmholtz Energy Model', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('Figures/helmholtz_energy.png', dpi=150)
    plt.show()
    
    # (b) Fisher Information Matrix analysis
    P_design = np.arange(0, 0.85, 0.05)
    S = np.column_stack([P_design**2, P_design**4, P_design**6])
    F = S.T @ S
    eigenvalues, eigenvectors = np.linalg.eig(F)
    
    print("=== Fisher Information Matrix Analysis ===")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Condition number: {np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues)):.2e}")
    
    # (c) Morris Screening
    # Parameter bounds (20% variation around nominal values)
    bounds = [
        (alpha_1 * 0.8, alpha_1 * 1.2),
        (alpha_11 * 0.8, alpha_11 * 1.2),
        (alpha_111 * 0.8, alpha_111 * 1.2)
    ]
    
    mu, mu_star, sigma = morris_screening(integrated_helmholtz, bounds, r=50)
    
    print("\n=== Morris Screening Results ===")
    print(f"Parameter   | mu       | mu*      | sigma")
    print(f"alpha_1     | {mu[0]:.4f} | {mu_star[0]:.4f} | {sigma[0]:.4f}")
    print(f"alpha_11    | {mu[1]:.4f} | {mu_star[1]:.4f} | {sigma[1]:.4f}")
    print(f"alpha_111   | {mu[2]:.4f} | {mu_star[2]:.4f} | {sigma[2]:.4f}")
    
    # (d) Sobol Indices
    S1, ST = sobol_indices(integrated_helmholtz, bounds, M=10000)
    
    print("\n=== Sobol Sensitivity Indices ===")
    print(f"Parameter   | S1 (First-order) | ST (Total-order)")
    print(f"alpha_1     | {S1[0]:.4f}          | {ST[0]:.4f}")
    print(f"alpha_11    | {S1[1]:.4f}          | {ST[1]:.4f}")
    print(f"alpha_111   | {S1[2]:.4f}          | {ST[2]:.4f}")
    print(f"\nSum of S1: {np.sum(S1):.4f} (should be close to 1 for additive model)")
    
    # (e) Compare distributions with fixed vs random alpha_111
    np.random.seed(42)
    M = 10000
    
    # All parameters random
    samples_all = np.random.rand(M, 3)
    for j, (a, b) in enumerate(bounds):
        samples_all[:, j] = samples_all[:, j] * (b - a) + a
    f_all = np.array([integrated_helmholtz(*s) for s in samples_all])
    
    # Only alpha_1 and alpha_11 random, alpha_111 fixed
    samples_fixed = np.random.rand(M, 2)
    samples_fixed[:, 0] = samples_fixed[:, 0] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
    samples_fixed[:, 1] = samples_fixed[:, 1] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
    f_fixed = np.array([integrated_helmholtz(s[0], s[1], alpha_111) for s in samples_fixed])
    
    # Plot comparison
    kde_all = gaussian_kde(f_all)
    kde_fixed = gaussian_kde(f_fixed)
    x_plot = np.linspace(min(f_all.min(), f_fixed.min()), max(f_all.max(), f_fixed.max()), 200)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, kde_all(x_plot), '-k', linewidth=2, label='All random')
    plt.plot(x_plot, kde_fixed(x_plot), '--r', linewidth=2, label=r'$\alpha_1, \alpha_{11}$ random')
    plt.xlabel('Integrated Energy', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Effect of Fixing alpha_111', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('Figures/sobol_comparison.png', dpi=150)
    plt.show()
    
    print("\n=== Conclusion ===")
    print("The Sobol indices show that alpha_1 has the largest influence on the output,")
    print("followed by alpha_11. alpha_111 has relatively small influence.")
    print("This is consistent with the Morris screening results.")


if __name__ == "__main__":
    main()
