"""
UQ Problem 8.5: Compute sensitivities of spring model

This script demonstrates computing sensitivities using:
1. Finite difference approximation
2. Complex-step approximation
3. Analytic solutions

The spring model: z'' + C*z' + K*z = 0
with initial conditions z(0) = 2, z'(0) = -C
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def spring_rhs(t, y, K, C):
    """ODE system for spring model."""
    z, z_dot = y
    return [z_dot, -C * z_dot - K * z]


def analytic_solution(t, K, C):
    """Analytic solution of the spring model."""
    omega = np.sqrt(K - C**2 / 4)
    return 2 * np.exp(-C * t / 2) * np.cos(omega * t)


def main():
    # Parameters
    T = 5  # Final time
    K = 20.5  # Spring constant
    C = 1.5  # Damping coefficient
    t_vals = np.arange(0, T + 0.05, 0.05)
    z_0 = 2
    z_dot_0 = -C
    
    # Define analytic solution function
    y = lambda K_val, C_val: analytic_solution(t_vals, K_val, C_val)
    
    # Finite difference step
    h_fd = 1e-8
    
    # dz/dK using Finite Difference
    z_K_finite = (y(K + h_fd, C) - y(K, C)) / h_fd
    
    # dz/dC using Finite Difference
    h_fd_C = 1e-7
    z_C_finite = (y(K, C + h_fd_C) - y(K, C)) / h_fd_C
    
    # Complex-step approximation
    h_cs = 1e-16
    z_K_complex = np.imag(analytic_solution(t_vals, complex(K, h_cs), C)) / h_cs
    z_C_complex = np.imag(analytic_solution(t_vals, K, complex(C, h_cs))) / h_cs
    
    # Complex-step with ODE solver for dz/dK
    def solve_spring_complex(K_val, C_val):
        sol = solve_ivp(
            lambda t, y: spring_rhs(t, y, K_val, C_val),
            [0, T],
            [z_0, z_dot_0],
            t_eval=t_vals,
            rtol=1e-8
        )
        return sol.y[0]
    
    z_K_ode = np.imag(solve_spring_complex(complex(K, h_cs), C)) / h_cs
    z_C_ode = np.imag(solve_spring_complex(K, complex(C, h_cs))) / h_cs
    
    # Analytic sensitivities
    omega = np.sqrt(K - C**2 / 4)
    y_K = np.exp(-C * t_vals / 2) * (-2 * t_vals / np.sqrt(4 * K - C**2)) * np.sin(omega * t_vals)
    y_C = np.exp(-C * t_vals / 2) * (
        (C * t_vals / np.sqrt(4 * K - C**2)) * np.sin(omega * t_vals) 
        - t_vals * np.cos(omega * t_vals)
    )
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot dz/dK
    axes[0].plot(t_vals, z_K_complex, '-k', linewidth=2, label='Complex Step')
    axes[0].plot(t_vals, y_K, '--c', linewidth=2, label='Analytic')
    axes[0].plot(t_vals, z_K_finite, 'r*', markersize=4, label='Finite Difference')
    axes[0].set_xlabel('Time', fontsize=14)
    axes[0].set_ylabel(r'$\partial z / \partial K$', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].set_title('Sensitivity with respect to K')
    axes[0].grid(True, alpha=0.3)
    
    # Plot dz/dC
    axes[1].plot(t_vals, z_C_complex, '-k', linewidth=2, label='Complex Step')
    axes[1].plot(t_vals, y_C, '--c', linewidth=2, label='Analytic')
    axes[1].plot(t_vals, z_C_finite, 'r*', markersize=4, label='Finite Difference')
    axes[1].set_xlabel('Time', fontsize=14)
    axes[1].set_ylabel(r'$\partial z / \partial C$', fontsize=14)
    axes[1].legend(loc='upper right')
    axes[1].set_title('Sensitivity with respect to C')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures/spring_sensitivities.png', dpi=150)
    plt.show()
    
    # Print verification
    print("Verification: Maximum differences between methods")
    print(f"  dz/dK: Complex vs Analytic = {np.max(np.abs(z_K_complex - y_K)):.2e}")
    print(f"  dz/dK: Finite Diff vs Analytic = {np.max(np.abs(z_K_finite - y_K)):.2e}")
    print(f"  dz/dC: Complex vs Analytic = {np.max(np.abs(z_C_complex - y_C)):.2e}")
    print(f"  dz/dC: Finite Diff vs Analytic = {np.max(np.abs(z_C_finite - y_C)):.2e}")


if __name__ == "__main__":
    main()
