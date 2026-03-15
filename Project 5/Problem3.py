"""
Project 5, Problem 3: Gaussian Process (Kriging) surrogate model

This script demonstrates:
1. Gaussian Process regression
2. Kernel selection and hyperparameter optimization
3. Uncertainty quantification with GP predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular


def rbf_kernel(X1, X2, length_scale, variance):
    """Radial Basis Function (squared exponential) kernel."""
    X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
    X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
    
    dist_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return variance * np.exp(-0.5 * dist_sq / length_scale**2)


class GaussianProcessRegressor:
    """Simple Gaussian Process regressor implementation."""
    
    def __init__(self, length_scale=1.0, variance=1.0, noise=1e-6):
        self.length_scale = length_scale
        self.variance = variance
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
    
    def fit(self, X, y):
        """Fit the GP model."""
        self.X_train = np.atleast_2d(X).T if X.ndim == 1 else X
        self.y_train = y
        
        K = rbf_kernel(self.X_train, self.X_train, self.length_scale, self.variance)
        K += self.noise * np.eye(len(X))
        
        self.L = cholesky(K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y, lower=True))
        
        return self
    
    def predict(self, X, return_std=False):
        """Predict mean and optionally standard deviation."""
        X = np.atleast_2d(X).T if X.ndim == 1 else X
        
        K_star = rbf_kernel(X, self.X_train, self.length_scale, self.variance)
        y_mean = K_star @ self.alpha
        
        if return_std:
            v = solve_triangular(self.L, K_star.T, lower=True)
            K_star_star = rbf_kernel(X, X, self.length_scale, self.variance)
            y_var = np.diag(K_star_star) - np.sum(v**2, axis=0)
            y_std = np.sqrt(np.maximum(y_var, 0))
            return y_mean, y_std
        
        return y_mean
    
    def negative_log_likelihood(self, params, X, y):
        """Compute negative log marginal likelihood."""
        length_scale, variance, noise = np.exp(params)
        
        X = np.atleast_2d(X).T if X.ndim == 1 else X
        n = len(y)
        
        K = rbf_kernel(X, X, length_scale, variance) + noise * np.eye(n)
        
        try:
            L = cholesky(K, lower=True)
            alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
            
            nll = 0.5 * y @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
            return nll
        except:
            return 1e10
    
    def optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters by minimizing negative log likelihood."""
        params0 = np.log([self.length_scale, self.variance, self.noise])
        
        result = minimize(
            lambda p: self.negative_log_likelihood(p, X, y),
            params0,
            method='L-BFGS-B',
            bounds=[(-5, 5), (-5, 5), (-10, -2)]
        )
        
        self.length_scale, self.variance, self.noise = np.exp(result.x)
        return self


def main():
    # Generate training data
    np.random.seed(42)
    
    # Target function
    f = lambda x: np.sin(3 * x) * np.exp(-0.5 * x)
    
    # Training points
    n_train = 10
    X_train = np.sort(np.random.uniform(0, 4, n_train))
    y_train = f(X_train) + 0.05 * np.random.randn(n_train)
    
    # Test points
    X_test = np.linspace(0, 4, 200)
    y_true = f(X_test)
    
    # Create and train GP
    gp = GaussianProcessRegressor(length_scale=1.0, variance=1.0, noise=0.01)
    gp.optimize_hyperparameters(X_train, y_train)
    gp.fit(X_train, y_train)
    
    # Predict
    y_pred, y_std = gp.predict(X_test, return_std=True)
    
    print("=== Gaussian Process Regression ===")
    print(f"Optimized hyperparameters:")
    print(f"  Length scale: {gp.length_scale:.4f}")
    print(f"  Variance: {gp.variance:.4f}")
    print(f"  Noise: {gp.noise:.6f}")
    
    # Compute RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    print(f"\nRMSE: {rmse:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GP prediction with uncertainty
    ax1 = axes[0]
    ax1.plot(X_test, y_true, 'b-', linewidth=2, label='True Function')
    ax1.plot(X_test, y_pred, 'r--', linewidth=2, label='GP Mean')
    ax1.fill_between(X_test, y_pred - 2*y_std, y_pred + 2*y_std, 
                     alpha=0.3, color='red', label='95% CI')
    ax1.plot(X_train, y_train, 'ko', markersize=8, label='Training Data')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Gaussian Process Regression', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uncertainty visualization
    ax2 = axes[1]
    ax2.plot(X_test, y_std, 'g-', linewidth=2)
    ax2.axhline(y=np.sqrt(gp.noise), color='gray', linestyle='--', 
                label=f'Noise level: {np.sqrt(gp.noise):.3f}')
    for x in X_train:
        ax2.axvline(x=x, color='red', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Predictive Std Dev', fontsize=12)
    ax2.set_title('Prediction Uncertainty', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures&Data/gaussian_process.png', dpi=150)
    plt.show()
    
    print("\n=== Key Observations ===")
    print("1. GP provides probabilistic predictions with uncertainty estimates")
    print("2. Uncertainty increases away from training data")
    print("3. Hyperparameters control smoothness and noise handling")


if __name__ == "__main__":
    main()
