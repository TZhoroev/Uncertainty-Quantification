"""
Project 3, Problem 1: Bayesian parameter estimation using MCMC

This script demonstrates:
1. Metropolis-Hastings MCMC algorithm
2. Delayed Rejection Adaptive Metropolis (DRAM)
3. Posterior distribution visualization
4. Comparison of MCMC methods

Heat model with parameters Q (Phi) and h.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Data
DATA = np.array([66.04, 60.04, 54.81, 50.42, 46.74, 43.66, 40.76, 38.49, 
                 36.42, 34.77, 33.18, 32.36, 31.56, 30.91, 30.56])
XDATA = np.array([10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66])
U_AMB = 22.28


def heat_model(x, Q, h, a=0.95, b=0.95, L=70.0, k=4.01):
    """Steady-state heat equation solution."""
    gamma = np.sqrt(2 * (a + b) * h / (a * b * k))
    f1 = np.exp(gamma * L) * (h + k * gamma)
    f2 = np.exp(-gamma * L) * (h - k * gamma)
    f3 = f1 / (f2 + f1)
    c1 = -Q * f3 / (k * gamma)
    c2 = Q / (k * gamma) + c1
    
    return c1 * np.exp(-gamma * x) + c2 * np.exp(gamma * x) + U_AMB


def log_likelihood(Q, h, sigma2):
    """Compute log-likelihood."""
    u_model = heat_model(XDATA, Q, h)
    residuals = DATA - u_model
    return -0.5 * np.sum(residuals**2) / sigma2


def sample_sigma2(residuals, n0=0.1, sigma02=0.1):
    """Sample measurement variance from inverse gamma posterior."""
    n = len(residuals)
    SS = np.sum(residuals**2)
    a = 0.5 * (n0 + n)
    b = 0.5 * (n0 * sigma02 + SS)
    return 1.0 / np.random.gamma(a, 1.0 / b)


def metropolis_hastings(n_iter, q_init, V_proposal, sigma2_init, 
                        update_sigma=True, n0=0.1, sigma02=0.1):
    """
    Metropolis-Hastings MCMC sampler.
    
    Parameters:
    -----------
    n_iter : int
        Number of iterations
    q_init : array
        Initial parameter values [Q, h]
    V_proposal : array
        Proposal covariance matrix
    sigma2_init : float
        Initial measurement variance
    update_sigma : bool
        Whether to update sigma2 during sampling
    """
    # Initialize
    q_current = np.array(q_init)
    sigma2 = sigma2_init
    R = np.linalg.cholesky(V_proposal).T
    
    # Storage
    samples = np.zeros((n_iter, 2))
    sigma2_samples = np.zeros(n_iter)
    accept_count = 0
    
    # Current log-likelihood
    u_current = heat_model(XDATA, q_current[0], q_current[1])
    SS_current = np.sum((DATA - u_current)**2)
    
    for i in range(n_iter):
        # Propose new parameters
        z = np.random.randn(2)
        q_proposed = q_current + R @ z
        
        # Compute proposed log-likelihood
        try:
            u_proposed = heat_model(XDATA, q_proposed[0], q_proposed[1])
            SS_proposed = np.sum((DATA - u_proposed)**2)
            
            # Accept/reject
            log_alpha = -0.5 * (SS_proposed - SS_current) / sigma2
            alpha = min(1.0, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                q_current = q_proposed
                SS_current = SS_proposed
                accept_count += 1
        except:
            pass  # Reject if model fails
        
        samples[i] = q_current
        
        # Update sigma2 if requested
        if update_sigma:
            residuals = DATA - heat_model(XDATA, q_current[0], q_current[1])
            sigma2 = sample_sigma2(residuals, n0, sigma02)
        
        sigma2_samples[i] = sigma2
    
    acceptance_rate = accept_count / n_iter
    return samples, sigma2_samples, acceptance_rate


def adaptive_metropolis(n_iter, q_init, V_init, sigma2_init, 
                        adapt_start=1000, adapt_interval=100):
    """
    Adaptive Metropolis algorithm that updates proposal covariance.
    """
    q_current = np.array(q_init)
    sigma2 = sigma2_init
    V_proposal = V_init.copy()
    
    samples = np.zeros((n_iter, 2))
    sigma2_samples = np.zeros(n_iter)
    accept_count = 0
    
    # Scaling factor for proposal
    s_d = 2.4**2 / 2  # Optimal scaling for 2D
    eps = 1e-6  # Small regularization
    
    u_current = heat_model(XDATA, q_current[0], q_current[1])
    SS_current = np.sum((DATA - u_current)**2)
    
    for i in range(n_iter):
        # Update proposal covariance adaptively
        if i >= adapt_start and i % adapt_interval == 0:
            V_proposal = s_d * np.cov(samples[:i].T) + eps * np.eye(2)
        
        try:
            R = np.linalg.cholesky(V_proposal).T
        except:
            R = np.linalg.cholesky(V_init).T
        
        # Propose
        z = np.random.randn(2)
        q_proposed = q_current + R @ z
        
        try:
            u_proposed = heat_model(XDATA, q_proposed[0], q_proposed[1])
            SS_proposed = np.sum((DATA - u_proposed)**2)
            
            log_alpha = -0.5 * (SS_proposed - SS_current) / sigma2
            alpha = min(1.0, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                q_current = q_proposed
                SS_current = SS_proposed
                accept_count += 1
        except:
            pass
        
        samples[i] = q_current
        
        # Update sigma2
        residuals = DATA - heat_model(XDATA, q_current[0], q_current[1])
        sigma2 = sample_sigma2(residuals)
        sigma2_samples[i] = sigma2
    
    acceptance_rate = accept_count / n_iter
    return samples, sigma2_samples, acceptance_rate


def main():
    # Initial values from OLS
    Q_init = -9.9265
    h_init = 0.0014
    n = len(DATA)
    p = 2
    
    # Initial model and residuals
    u_init = heat_model(XDATA, Q_init, h_init)
    residuals = DATA - u_init
    sigma2_init = np.sum(residuals**2) / (n - p)
    
    # Compute initial covariance from sensitivity analysis
    # (Using approximate values)
    V_proposal = np.array([[0.01, 0], [0, 1e-9]])
    
    print("=== Running Metropolis-Hastings MCMC ===")
    N = 100000
    
    samples_mh, sigma2_mh, acc_rate_mh = metropolis_hastings(
        N, [Q_init, h_init], V_proposal, sigma2_init
    )
    
    print(f"MH Acceptance rate: {acc_rate_mh:.3f}")
    
    print("\n=== Running Adaptive Metropolis ===")
    samples_am, sigma2_am, acc_rate_am = adaptive_metropolis(
        N, [Q_init, h_init], V_proposal, sigma2_init
    )
    
    print(f"AM Acceptance rate: {acc_rate_am:.3f}")
    
    # Extract chains
    Q_mh = samples_mh[:, 0]
    h_mh = samples_mh[:, 1]
    Q_am = samples_am[:, 0]
    h_am = samples_am[:, 1]
    
    # Compute densities using KDE
    kde_Q_mh = gaussian_kde(Q_mh)
    kde_h_mh = gaussian_kde(h_mh)
    kde_Q_am = gaussian_kde(Q_am)
    kde_h_am = gaussian_kde(h_am)
    
    # Plot ranges
    Q_range = np.linspace(Q_mh.min() - 0.1, Q_mh.max() + 0.1, 200)
    h_range = np.linspace(h_mh.min() - 1e-5, h_mh.max() + 1e-5, 200)
    
    # Plotting
    fig = plt.figure(figsize=(16, 12))
    
    # Trace plots
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(Q_mh, linewidth=0.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Q')
    ax1.set_title('MH: Trace of Q')
    
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(h_mh, linewidth=0.5)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('h')
    ax2.set_title('MH: Trace of h')
    
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(sigma2_mh, linewidth=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel(r'$\sigma^2$')
    ax3.set_title(r'MH: Trace of $\sigma^2$')
    
    # Posterior densities
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(Q_range, kde_Q_mh(Q_range), 'b-', linewidth=2, label='MH')
    ax4.plot(Q_range, kde_Q_am(Q_range), 'r--', linewidth=2, label='AM')
    ax4.set_xlabel('Q')
    ax4.set_ylabel('Density')
    ax4.set_title('Posterior of Q')
    ax4.legend()
    
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.plot(h_range, kde_h_mh(h_range), 'b-', linewidth=2, label='MH')
    ax5.plot(h_range, kde_h_am(h_range), 'r--', linewidth=2, label='AM')
    ax5.set_xlabel('h')
    ax5.set_ylabel('Density')
    ax5.set_title('Posterior of h')
    ax5.legend()
    
    # Joint posterior
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.scatter(Q_mh[::100], h_mh[::100], alpha=0.3, s=5)
    ax6.set_xlabel('Q')
    ax6.set_ylabel('h')
    ax6.set_title('Joint Posterior (MH)')
    
    # Adaptive traces
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.plot(Q_am, linewidth=0.5)
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Q')
    ax7.set_title('AM: Trace of Q')
    
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.plot(h_am, linewidth=0.5)
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('h')
    ax8.set_title('AM: Trace of h')
    
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.scatter(Q_am[::100], h_am[::100], alpha=0.3, s=5)
    ax9.set_xlabel('Q')
    ax9.set_ylabel('h')
    ax9.set_title('Joint Posterior (AM)')
    
    plt.tight_layout()
    plt.savefig('Figures&Data/mcmc_comparison.png', dpi=150)
    plt.show()
    
    # Summary statistics
    burn_in = 10000
    
    print("\n=== Posterior Summary (after burn-in) ===")
    print("\nMetropolis-Hastings:")
    print(f"  Q: mean = {np.mean(Q_mh[burn_in:]):.4f}, std = {np.std(Q_mh[burn_in:]):.4f}")
    print(f"  h: mean = {np.mean(h_mh[burn_in:]):.6f}, std = {np.std(h_mh[burn_in:]):.6f}")
    print(f"  sigma2: mean = {np.mean(sigma2_mh[burn_in:]):.4f}")
    
    print("\nAdaptive Metropolis:")
    print(f"  Q: mean = {np.mean(Q_am[burn_in:]):.4f}, std = {np.std(Q_am[burn_in:]):.4f}")
    print(f"  h: mean = {np.mean(h_am[burn_in:]):.6f}, std = {np.std(h_am[burn_in:]):.6f}")
    print(f"  sigma2: mean = {np.mean(sigma2_am[burn_in:]):.4f}")
    
    # Credible intervals
    print("\n=== 95% Credible Intervals (MH) ===")
    print(f"  Q: [{np.percentile(Q_mh[burn_in:], 2.5):.4f}, {np.percentile(Q_mh[burn_in:], 97.5):.4f}]")
    print(f"  h: [{np.percentile(h_mh[burn_in:], 2.5):.6f}, {np.percentile(h_mh[burn_in:], 97.5):.6f}]")


if __name__ == "__main__":
    main()
