"""
Project 4, Problem 1: Frequentist confidence and prediction intervals

This script demonstrates:
1. Computing confidence intervals for model predictions
2. Computing prediction intervals
3. Uncertainty propagation in calibration and extrapolation domains

Height-weight model: weight = theta1 + theta2*(height/12) + theta3*(height/12)^2
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def main():
    # Data
    x_data = np.arange(58, 73)  # Heights in inches
    n = len(x_data)
    p = 3  # Number of parameters
    
    Y_data = np.array([115, 117, 120, 123, 126, 129, 132, 135, 139, 142, 146, 150, 154, 159, 164])
    
    # Given parameter estimates (from Example 11.7 in textbook)
    theta = np.array([261.88, -88.18, 11.96])
    
    # Design matrix
    X = np.column_stack([np.ones(n), x_data / 12, (x_data / 12)**2])
    
    # Estimated variance
    sigma2 = 0.1491
    sigma = np.sqrt(sigma2)
    
    # Covariance matrix of parameters
    XtX_inv = np.linalg.inv(X.T @ X)
    V_theta = sigma2 * XtX_inv
    
    # Model predictions at data points
    y_pred = X @ theta
    
    # Covariance of predictions
    V_y = X @ V_theta @ X.T + sigma2 * np.eye(n)
    sd_conf = 2 * np.sqrt(np.diag(X @ V_theta @ X.T))  # 2-sigma confidence band
    
    # Prediction intervals for new observations
    # Test points (calibration domain)
    x_test_cal = np.arange(58, 73, 0.5)
    N_cal = len(x_test_cal)
    X_test_cal = np.column_stack([np.ones(N_cal), x_test_cal / 12, (x_test_cal / 12)**2])
    y_test_cal = X_test_cal @ theta
    
    # t-critical value for 95% intervals
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, n - p)
    
    # Prediction intervals (calibration domain)
    pred_upper_cal = np.zeros(N_cal)
    pred_lower_cal = np.zeros(N_cal)
    
    for j in range(N_cal):
        se_pred = sigma * np.sqrt(1 + X_test_cal[j] @ XtX_inv @ X_test_cal[j])
        pred_upper_cal[j] = y_test_cal[j] + t_crit * se_pred
        pred_lower_cal[j] = y_test_cal[j] - t_crit * se_pred
    
    # Extrapolation domain
    x_test_ext = np.arange(50, 81, 0.5)
    N_ext = len(x_test_ext)
    X_test_ext = np.column_stack([np.ones(N_ext), x_test_ext / 12, (x_test_ext / 12)**2])
    y_test_ext = X_test_ext @ theta
    
    pred_upper_ext = np.zeros(N_ext)
    pred_lower_ext = np.zeros(N_ext)
    
    for j in range(N_ext):
        se_pred = sigma * np.sqrt(1 + X_test_ext[j] @ XtX_inv @ X_test_ext[j])
        pred_upper_ext[j] = y_test_ext[j] + t_crit * se_pred
        pred_lower_ext[j] = y_test_ext[j] - t_crit * se_pred
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calibration domain
    ax1 = axes[0]
    ax1.plot(x_test_cal, y_test_cal, '-k', linewidth=2, label='Mean Response')
    ax1.plot(x_data, Y_data, '*m', markersize=10, label='Data')
    ax1.fill_between(x_data, y_pred - sd_conf, y_pred + sd_conf, 
                     alpha=0.4, color='blue', label=r'$2\sigma$ Confidence')
    ax1.fill_between(x_test_cal, pred_lower_cal, pred_upper_cal, 
                     alpha=0.3, color='gray', edgecolor='red', label='95% Prediction')
    ax1.set_xlabel('Height (in)', fontsize=14)
    ax1.set_ylabel('Weight (lbs)', fontsize=14)
    ax1.set_title('Calibration Domain', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Extrapolation domain
    ax2 = axes[1]
    ax2.plot(x_test_ext, y_test_ext, '-k', linewidth=2, label='Mean Response')
    ax2.plot(x_data, Y_data, '*m', markersize=10, label='Data')
    ax2.fill_between(x_data, y_pred - sd_conf, y_pred + sd_conf, 
                     alpha=0.4, color='blue', label=r'$2\sigma$ Confidence')
    ax2.fill_between(x_test_ext, pred_lower_ext, pred_upper_ext, 
                     alpha=0.3, color='gray', edgecolor='red', label='95% Prediction')
    ax2.set_xlabel('Height (in)', fontsize=14)
    ax2.set_ylabel('Weight (lbs)', fontsize=14)
    ax2.set_title('Extrapolation Domain', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures & Data/frequentist_intervals.png', dpi=150)
    plt.show()
    
    # Print summary
    print("=== Model Summary ===")
    print(f"Parameters: theta = {theta}")
    print(f"Estimated sigma: {sigma:.4f}")
    print(f"t-critical (95%): {t_crit:.4f}")
    
    print("\n=== Prediction Interval Width ===")
    print(f"At x=65 (center): {pred_upper_cal[14] - pred_lower_cal[14]:.2f} lbs")
    print(f"At x=58 (edge): {pred_upper_cal[0] - pred_lower_cal[0]:.2f} lbs")
    print(f"At x=50 (extrapolation): {pred_upper_ext[0] - pred_lower_ext[0]:.2f} lbs")
    print(f"At x=80 (extrapolation): {pred_upper_ext[-1] - pred_lower_ext[-1]:.2f} lbs")


if __name__ == "__main__":
    main()
