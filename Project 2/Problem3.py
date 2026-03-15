"""
Project 2, Problem 3: Parameter distributions for SIR model

This script demonstrates:
1. Parameter estimation for SIR epidemic model
2. Distribution estimation using OLS
3. Asymptotic covariance analysis
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats


def sir_model(t, y, gamma, k, delta, r, N=1000):
    """SIR model differential equations."""
    S, I, R = y
    
    dS = delta * (N - S) - gamma * k * I * S
    dI = gamma * k * I * S - (r + delta) * I
    dR = r * I - delta * R
    
    return [dS, dI, dR]


def simulate_sir(params, t_vals, y0, N=1000):
    """Simulate SIR model and return solution."""
    gamma, k, delta, r = params
    
    sol = solve_ivp(
        lambda t, y: sir_model(t, y, gamma, k, delta, r, N),
        [t_vals[0], t_vals[-1]],
        y0,
        t_eval=t_vals,
        method='RK45',
        rtol=1e-8
    )
    
    return sol.y


def objective(params, t_data, data, y0, component=1):
    """
    Objective function for parameter estimation.
    
    Parameters:
    -----------
    component : int
        0 = S, 1 = I, 2 = R (which compartment to fit)
    """
    try:
        y = simulate_sir(params, t_data, y0)
        residuals = data - y[component]
        return np.sum(residuals**2)
    except:
        return 1e10


def compute_sensitivities_sir(params, t_data, y0, h=1e-8):
    """Compute sensitivities using finite differences."""
    n_params = len(params)
    n_time = len(t_data)
    
    y_base = simulate_sir(params, t_data, y0)
    
    sens = np.zeros((3, n_time, n_params))  # 3 states x time x parameters
    
    for j in range(n_params):
        params_p = params.copy()
        params_p[j] += h
        y_p = simulate_sir(params_p, t_data, y0)
        
        for state in range(3):
            sens[state, :, j] = (y_p[state] - y_base[state]) / h
    
    return sens


def main():
    # Time parameters
    tf = 5.0
    dt = 0.1
    t_data = np.arange(0, tf + dt, dt)
    
    # True parameters and initial conditions
    N = 1000
    gamma_true = 0.2
    k_true = 0.1
    delta_true = 0.15
    r_true = 0.6
    params_true = [gamma_true, k_true, delta_true, r_true]
    
    S0, I0, R0 = 900, 100, 0
    y0 = [S0, I0, R0]
    
    # Generate synthetic data with noise
    np.random.seed(42)
    y_true = simulate_sir(params_true, t_data, y0)
    
    sigma_noise = 10.0
    I_data = y_true[1] + np.random.normal(0, sigma_noise, len(t_data))
    
    # Initial parameter guess
    params_init = [0.15, 0.08, 0.1, 0.5]
    
    # Optimize parameters
    result = minimize(
        lambda p: objective(p, t_data, I_data, y0, component=1),
        params_init,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8}
    )
    
    params_est = result.x
    
    print("=== Parameter Estimation Results ===")
    param_names = ['gamma', 'k', 'delta', 'r']
    for i, name in enumerate(param_names):
        print(f"{name}: {params_est[i]:.4f} (true: {params_true[i]})")
    
    # Compute model predictions
    y_est = simulate_sir(params_est, t_data, y0)
    
    # Compute residuals and variance
    residuals = I_data - y_est[1]
    n = len(t_data)
    p = len(params_est)
    sigma2_est = np.sum(residuals**2) / (n - p)
    
    print(f"\nEstimated measurement std: {np.sqrt(sigma2_est):.4f} (true: {sigma_noise})")
    
    # Compute sensitivities
    sens = compute_sensitivities_sir(params_est, t_data, y0)
    sens_I = sens[1]  # Sensitivities of I compartment
    
    # Fisher information matrix
    F = sens_I.T @ sens_I
    
    # Covariance matrix
    try:
        V = sigma2_est * np.linalg.inv(F)
        
        print("\n=== Parameter Covariance Matrix ===")
        print(V)
        
        # Standard errors
        se = np.sqrt(np.diag(V))
        
        # 95% confidence intervals
        t_val = stats.t.ppf(0.975, n - p)
        
        print("\n=== 95% Confidence Intervals ===")
        for i, name in enumerate(param_names):
            ci = [params_est[i] - t_val * se[i], params_est[i] + t_val * se[i]]
            print(f"{name}: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
    except np.linalg.LinAlgError:
        print("\nWarning: Fisher matrix is singular. Parameters may not be identifiable.")
        V = None
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # SIR dynamics
    axes[0, 0].plot(t_data, y_true[0], 'b-', linewidth=2, label='S (true)')
    axes[0, 0].plot(t_data, y_true[1], 'r-', linewidth=2, label='I (true)')
    axes[0, 0].plot(t_data, y_true[2], 'g-', linewidth=2, label='R (true)')
    axes[0, 0].plot(t_data, I_data, 'ko', markersize=4, alpha=0.5, label='I (data)')
    axes[0, 0].set_xlabel('Time', fontsize=12)
    axes[0, 0].set_ylabel('Population', fontsize=12)
    axes[0, 0].set_title('SIR Model - True Dynamics', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Model fit
    axes[0, 1].plot(t_data, y_est[0], 'b--', linewidth=2, label='S (est)')
    axes[0, 1].plot(t_data, y_est[1], 'r--', linewidth=2, label='I (est)')
    axes[0, 1].plot(t_data, y_est[2], 'g--', linewidth=2, label='R (est)')
    axes[0, 1].plot(t_data, I_data, 'ko', markersize=4, alpha=0.5, label='I (data)')
    axes[0, 1].set_xlabel('Time', fontsize=12)
    axes[0, 1].set_ylabel('Population', fontsize=12)
    axes[0, 1].set_title('SIR Model - Estimated', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals
    axes[1, 0].plot(t_data, residuals, 'bo', markersize=6)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Residuals', fontsize=12)
    axes[1, 0].set_title('Residuals', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sensitivities
    for i, name in enumerate(param_names):
        axes[1, 1].plot(t_data, sens_I[:, i], linewidth=2, label=name)
    axes[1, 1].set_xlabel('Time', fontsize=12)
    axes[1, 1].set_ylabel('Sensitivity of I', fontsize=12)
    axes[1, 1].set_title('Sensitivities', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures& Data/problem3_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
