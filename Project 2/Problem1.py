"""
Project 2, Problem 1: Parameter estimation for heat model using constrained optimization

This script demonstrates:
1. Non-linear constrained optimization for parameter estimation
2. Computing sensitivities analytically and using finite differences
3. Confidence interval construction using Fisher information
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats


# Global data
DATA_COP = np.array([66.04, 60.04, 54.81, 50.42, 46.74, 43.66, 40.76, 38.49, 
                     36.42, 34.77, 33.18, 32.36, 31.56, 30.91, 30.56])
XDATA = np.array([10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66])
U_AMB = 22.28


def heat_model(x, h, Q, a=0.95, b=0.95, L=70.0, k=4.01, u_amb=22.28):
    """
    Steady-state heat equation solution for a rod with convection.
    
    Parameters:
    -----------
    x : array-like
        Spatial positions
    h : float
        Heat transfer coefficient
    Q : float
        Heat flux parameter (Phi)
    """
    gamma = np.sqrt(2 * (a + b) * h / (a * b * k))
    f1 = np.exp(gamma * L) * (h + k * gamma)
    f2 = np.exp(-gamma * L) * (h - k * gamma)
    f3 = f1 / (f2 + f1)
    c1 = -Q * f3 / (k * gamma)
    c2 = Q / (k * gamma) + c1
    
    return c1 * np.exp(-gamma * x) + c2 * np.exp(gamma * x) + u_amb


def objective(params):
    """Objective function: sum of squared residuals."""
    h, Q = params
    u_model = heat_model(XDATA, h, Q)
    residuals = DATA_COP - u_model
    return np.sum(residuals**2)


def compute_sensitivities(x, h, Q, a=0.95, b=0.95, L=70.0, k=4.01):
    """
    Compute analytic sensitivities of the heat model.
    
    Returns sensitivity with respect to Q and h.
    """
    gamma = np.sqrt(2 * (a + b) * h / (a * b * k))
    gamma_h = (1 / (2 * h)) * gamma
    
    f1 = np.exp(gamma * L) * (h + k * gamma)
    f2 = np.exp(-gamma * L) * (h - k * gamma)
    f3 = f1 / (f2 + f1)
    
    f1_h = np.exp(gamma * L) * (gamma_h * L * (h + k * gamma) + 1 + k * gamma_h)
    f2_h = np.exp(-gamma * L) * (-gamma_h * L * (h - k * gamma) + 1 - k * gamma_h)
    
    c1 = -Q * f3 / (k * gamma)
    c2 = Q / (k * gamma) + c1
    
    f4 = Q / (k * gamma**2)
    den2 = (f1 + f2)**2
    f3_h = (f1_h * (f1 + f2) - f1 * (f1_h + f2_h)) / den2
    
    c1_h = f4 * gamma_h * f3 - (Q / (k * gamma)) * f3_h
    c2_h = -f4 * gamma_h + c1_h
    c1_Q = -(1 / (k * gamma)) * f3
    c2_Q = (1 / (k * gamma)) + c1_Q
    
    # Sensitivity with respect to Q
    du_dQ = c1_Q * np.exp(-gamma * x) + c2_Q * np.exp(gamma * x)
    
    # Sensitivity with respect to h
    du_dh = (c1_h * np.exp(-gamma * x) + c2_h * np.exp(gamma * x) + 
             gamma_h * x * (-c1 * np.exp(-gamma * x) + c2 * np.exp(gamma * x)))
    
    return du_dQ, du_dh


def compute_sensitivities_fd(x, h, Q, dh=1e-6, dQ=1e-4):
    """Compute sensitivities using finite differences."""
    u_base = heat_model(x, h, Q)
    
    du_dh = (heat_model(x, h + dh, Q) - u_base) / dh
    du_dQ = (heat_model(x, h, Q + dQ) - u_base) / dQ
    
    return du_dQ, du_dh


def main():
    # Problem parameters
    a, b, L, k = 0.95, 0.95, 70.0, 4.01
    n = len(DATA_COP)  # Number of measurements
    p = 2  # Number of parameters
    
    # Initial guess
    h_init = 0.00183
    Q_init = -15.93
    
    # Optimize using scipy.optimize.minimize
    result = minimize(
        objective,
        [h_init, Q_init],
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    h_opt, Q_opt = result.x
    print("=== Optimization Results ===")
    print(f"Optimal h: {h_opt:.6f}")
    print(f"Optimal Q (Phi): {Q_opt:.4f}")
    print(f"Final SSE: {result.fun:.6f}")
    
    # Compute model predictions and residuals
    xvals = np.linspace(10, 70, 200)
    u_model = heat_model(xvals, h_opt, Q_opt)
    u_data = heat_model(XDATA, h_opt, Q_opt)
    residuals = DATA_COP - u_data
    
    # Compute sensitivities
    du_dQ, du_dh = compute_sensitivities(XDATA, h_opt, Q_opt)
    du_dQ_fd, du_dh_fd = compute_sensitivities_fd(XDATA, h_opt, Q_opt)
    
    # Build sensitivity matrix
    sens_mat = np.vstack([du_dQ, du_dh])
    
    # Estimate measurement variance
    sigma2 = np.sum(residuals**2) / (n - p)
    print(f"\nEstimated measurement variance: {sigma2:.6f}")
    
    # Covariance matrix of parameters
    V = sigma2 * np.linalg.inv(sens_mat @ sens_mat.T)
    print(f"\nParameter covariance matrix:")
    print(V)
    
    # 95% confidence intervals
    t_val = stats.t.ppf(0.975, n - p)  # Two-tailed t-value
    
    ci_Q = [Q_opt - t_val * np.sqrt(V[0, 0]), Q_opt + t_val * np.sqrt(V[0, 0])]
    ci_h = [h_opt - t_val * np.sqrt(V[1, 1]), h_opt + t_val * np.sqrt(V[1, 1])]
    
    print(f"\n=== 95% Confidence Intervals ===")
    print(f"Q (Phi): [{ci_Q[0]:.4f}, {ci_Q[1]:.4f}]")
    print(f"h: [{ci_h[0]:.6f}, {ci_h[1]:.6f}]")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Model fit
    axes[0, 0].plot(xvals, u_model, '-b', linewidth=2, label='Model')
    axes[0, 0].plot(XDATA, DATA_COP, 'ro', markersize=8, label='Data')
    axes[0, 0].set_xlabel('Distance (cm)', fontsize=12)
    axes[0, 0].set_ylabel('Temperature (°C)', fontsize=12)
    axes[0, 0].set_title('Model Fit', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals
    axes[0, 1].plot(XDATA, residuals, 'bo', markersize=8)
    axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[0, 1].set_xlabel('Distance (cm)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals (°C)', fontsize=12)
    axes[0, 1].set_title('Residuals', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sensitivity w.r.t Q
    axes[1, 0].plot(XDATA, du_dQ, 'bo-', linewidth=2, markersize=6, label='Analytic')
    axes[1, 0].plot(XDATA, du_dQ_fd, 'rx', markersize=10, label='Finite Diff')
    axes[1, 0].set_xlabel('Distance (cm)', fontsize=12)
    axes[1, 0].set_ylabel(r'$\partial u / \partial \Phi$', fontsize=12)
    axes[1, 0].set_title('Sensitivity w.r.t. Phi', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivity w.r.t h
    axes[1, 1].plot(XDATA, du_dh, 'bo-', linewidth=2, markersize=6, label='Analytic')
    axes[1, 1].plot(XDATA, du_dh_fd, 'rx', markersize=10, label='Finite Diff')
    axes[1, 1].set_xlabel('Distance (cm)', fontsize=12)
    axes[1, 1].set_ylabel(r'$\partial u / \partial h$', fontsize=12)
    axes[1, 1].set_title('Sensitivity w.r.t. h', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures& Data/problem1_results.png', dpi=150)
    plt.show()
    
    # Q-Q plot for residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals', fontsize=14)
    plt.savefig('Figures& Data/problem1_qqplot.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
