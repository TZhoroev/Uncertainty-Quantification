"""
Project 2, Problem 2: OLS estimation for Helmholtz energy model

This script demonstrates:
1. Ordinary Least Squares (OLS) parameter estimation
2. Variance estimation
3. Confidence interval construction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def helmholtz_energy(P, alpha_1, alpha_11, alpha_111):
    """Helmholtz energy function."""
    return alpha_1 * P**2 + alpha_11 * P**4 + alpha_111 * P**6


def main():
    # Generate synthetic data with noise
    np.random.seed(42)
    
    # True parameters
    alpha_1_true = -389.4
    alpha_11_true = 761.3
    alpha_111_true = 61.5
    
    # Observation points
    P_obs = np.linspace(0.1, 0.8, 20)
    n = len(P_obs)
    p = 3  # Number of parameters
    
    # Generate noisy observations
    sigma_true = 5.0
    y_true = helmholtz_energy(P_obs, alpha_1_true, alpha_11_true, alpha_111_true)
    y_obs = y_true + np.random.normal(0, sigma_true, n)
    
    # Design matrix for linear regression
    X = np.column_stack([P_obs**2, P_obs**4, P_obs**6])
    
    # OLS estimation: theta = (X'X)^{-1} X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    theta_ols = XtX_inv @ X.T @ y_obs
    
    alpha_1_est, alpha_11_est, alpha_111_est = theta_ols
    
    print("=== OLS Parameter Estimates ===")
    print(f"alpha_1:   {alpha_1_est:.4f} (true: {alpha_1_true})")
    print(f"alpha_11:  {alpha_11_est:.4f} (true: {alpha_11_true})")
    print(f"alpha_111: {alpha_111_est:.4f} (true: {alpha_111_true})")
    
    # Compute residuals and variance estimate
    y_pred = X @ theta_ols
    residuals = y_obs - y_pred
    sigma2_est = np.sum(residuals**2) / (n - p)
    sigma_est = np.sqrt(sigma2_est)
    
    print(f"\nEstimated noise std: {sigma_est:.4f} (true: {sigma_true})")
    
    # Covariance matrix of parameters
    V_theta = sigma2_est * XtX_inv
    
    print("\nParameter covariance matrix:")
    print(V_theta)
    
    # 95% confidence intervals
    t_val = stats.t.ppf(0.975, n - p)
    
    se = np.sqrt(np.diag(V_theta))
    ci_alpha_1 = [alpha_1_est - t_val * se[0], alpha_1_est + t_val * se[0]]
    ci_alpha_11 = [alpha_11_est - t_val * se[1], alpha_11_est + t_val * se[1]]
    ci_alpha_111 = [alpha_111_est - t_val * se[2], alpha_111_est + t_val * se[2]]
    
    print("\n=== 95% Confidence Intervals ===")
    print(f"alpha_1:   [{ci_alpha_1[0]:.4f}, {ci_alpha_1[1]:.4f}]")
    print(f"alpha_11:  [{ci_alpha_11[0]:.4f}, {ci_alpha_11[1]:.4f}]")
    print(f"alpha_111: [{ci_alpha_111[0]:.4f}, {ci_alpha_111[1]:.4f}]")
    
    # Check if true values fall within confidence intervals
    print("\n=== Coverage Check ===")
    print(f"alpha_1 in CI: {ci_alpha_1[0] <= alpha_1_true <= ci_alpha_1[1]}")
    print(f"alpha_11 in CI: {ci_alpha_11[0] <= alpha_11_true <= ci_alpha_11[1]}")
    print(f"alpha_111 in CI: {ci_alpha_111[0] <= alpha_111_true <= ci_alpha_111[1]}")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Model fit
    P_plot = np.linspace(0, 0.8, 100)
    y_fit = helmholtz_energy(P_plot, alpha_1_est, alpha_11_est, alpha_111_est)
    y_true_plot = helmholtz_energy(P_plot, alpha_1_true, alpha_11_true, alpha_111_true)
    
    axes[0].plot(P_plot, y_true_plot, 'b-', linewidth=2, label='True')
    axes[0].plot(P_plot, y_fit, 'r--', linewidth=2, label='Estimated')
    axes[0].plot(P_obs, y_obs, 'ko', markersize=6, label='Data')
    axes[0].set_xlabel('Polarization P', fontsize=12)
    axes[0].set_ylabel('Helmholtz Energy', fontsize=12)
    axes[0].set_title('Model Fit', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    axes[1].plot(P_obs, residuals, 'bo', markersize=8)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Polarization P', fontsize=12)
    axes[1].set_ylabel('Residuals', fontsize=12)
    axes[1].set_title('Residuals', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('Figures& Data/problem2_results.png', dpi=150)
    plt.show()
    
    # Fisher Information Matrix analysis
    print("\n=== Fisher Information Analysis ===")
    F = X.T @ X / sigma2_est
    eigenvalues = np.linalg.eigvalsh(F)
    print(f"Eigenvalues of Fisher matrix: {eigenvalues}")
    print(f"Condition number: {np.max(eigenvalues) / np.min(eigenvalues):.2e}")


if __name__ == "__main__":
    main()
