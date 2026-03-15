"""
UQ Problem 8.8: Compute sensitivities of SIR model and find identifiable subset

This script demonstrates:
1. Computing sensitivities using sensitivity equations
2. Computing sensitivities using complex-step approximation
3. Identifying identifiable parameter subsets using eigenvalue decomposition

SIR Model:
  dS/dt = delta*(N-S) - gamma*k*I*S
  dI/dt = gamma*k*I*S - (r + delta)*I
  dR/dt = r*I - delta*R
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def sir_rhs(t, y, params):
    """SIR model differential equations."""
    N = 1000
    gamma, k, delta, r = params
    S, I, R = y
    
    dS = delta * (N - S) - gamma * k * I * S
    dI = gamma * k * I * S - (r + delta) * I
    dR = r * I - delta * R
    
    return [dS, dI, dR]


def sir_sensitivity_rhs(t, y, params):
    """SIR model with sensitivity equations."""
    N = 1000
    gamma, k, delta, r = params
    
    # State variables
    S, I, R = y[0], y[1], y[2]
    
    # Sensitivity variables (12 total: 3 states x 4 parameters)
    S_gamma, I_gamma, R_gamma = y[3], y[4], y[5]
    S_k, I_k, R_k = y[6], y[7], y[8]
    S_delta, I_delta, R_delta = y[9], y[10], y[11]
    S_r, I_r, R_r = y[12], y[13], y[14]
    
    # State equations
    dS = delta * (N - S) - gamma * k * I * S
    dI = gamma * k * I * S - (r + delta) * I
    dR = r * I - delta * R
    
    # Jacobian matrix
    J = np.array([
        [-(delta + gamma * k * I), -gamma * k * S, 0],
        [gamma * k * I, gamma * k * S - (r + delta), 0],
        [0, r, -delta]
    ])
    
    # Gradient terms for each parameter
    # Sensitivity w.r.t gamma
    grad_gamma = np.array([-k * I * S, k * I * S, 0])
    sens_gamma = J @ np.array([S_gamma, I_gamma, R_gamma]) + grad_gamma
    
    # Sensitivity w.r.t k
    grad_k = np.array([-gamma * I * S, gamma * I * S, 0])
    sens_k = J @ np.array([S_k, I_k, R_k]) + grad_k
    
    # Sensitivity w.r.t delta
    grad_delta = np.array([N - S, -I, -R])
    sens_delta = J @ np.array([S_delta, I_delta, R_delta]) + grad_delta
    
    # Sensitivity w.r.t r
    grad_r = np.array([0, -I, I])
    sens_r = J @ np.array([S_r, I_r, R_r]) + grad_r
    
    return [dS, dI, dR] + list(sens_gamma) + list(sens_k) + list(sens_delta) + list(sens_r)


def pss_eig(sens_mat, eta):
    """
    Parameter Subset Selection using eigenvalue decomposition.
    
    Returns identifiable and unidentifiable parameter indices.
    """
    n, p = sens_mat.shape
    identifiable = list(range(p))
    sens_work = sens_mat.copy()
    
    for _ in range(p):
        F = sens_work.T @ sens_work
        eigenvalues, eigenvectors = np.linalg.eig(F)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        if np.abs(eigenvalues[0]) > eta:
            break
        else:
            # Find parameter with largest contribution to smallest eigenvector
            max_idx = np.argmax(np.abs(eigenvectors[:, 0]))
            sens_work = np.delete(sens_work, max_idx, axis=1)
            identifiable.pop(max_idx)
    
    unidentifiable = [i for i in range(p) if i not in identifiable]
    return identifiable, unidentifiable


def main():
    # Time parameters
    tf = 5
    dt = 0.1
    t_vals = np.arange(0, tf + dt, dt)
    
    # Initial conditions
    S0, I0, R0 = 900, 100, 0
    
    # Parameters
    gamma, k, delta, r = 0.2, 0.1, 0.15, 0.6
    params = [gamma, k, delta, r]
    
    # Initial conditions for sensitivity equations (all zeros)
    y0_sens = [S0, I0, R0] + [0] * 12
    
    # Solve coupled system
    sol_sens = solve_ivp(
        lambda t, y: sir_sensitivity_rhs(t, y, params),
        [0, tf],
        y0_sens,
        t_eval=t_vals,
        rtol=1e-8
    )
    
    # Extract sensitivities from sensitivity equations
    S_gamma_sen, I_gamma_sen, R_gamma_sen = sol_sens.y[3], sol_sens.y[4], sol_sens.y[5]
    S_k_sen, I_k_sen, R_k_sen = sol_sens.y[6], sol_sens.y[7], sol_sens.y[8]
    S_delta_sen, I_delta_sen, R_delta_sen = sol_sens.y[9], sol_sens.y[10], sol_sens.y[11]
    S_r_sen, I_r_sen, R_r_sen = sol_sens.y[12], sol_sens.y[13], sol_sens.y[14]
    
    # Complex-step approximation
    h = 1e-16
    y0 = [S0, I0, R0]
    
    def solve_sir_complex(params_complex):
        sol = solve_ivp(
            lambda t, y: sir_rhs(t, y, params_complex),
            [0, tf],
            y0,
            t_eval=t_vals,
            rtol=1e-8
        )
        return sol.y
    
    # Sensitivity w.r.t gamma
    params_gamma = [complex(gamma, h), k, delta, r]
    Y_gamma = solve_sir_complex(params_gamma)
    S_gamma = np.imag(Y_gamma[0]) / h
    I_gamma = np.imag(Y_gamma[1]) / h
    R_gamma = np.imag(Y_gamma[2]) / h
    
    # Sensitivity w.r.t k
    params_k = [gamma, complex(k, h), delta, r]
    Y_k = solve_sir_complex(params_k)
    S_k = np.imag(Y_k[0]) / h
    I_k = np.imag(Y_k[1]) / h
    R_k = np.imag(Y_k[2]) / h
    
    # Sensitivity w.r.t delta
    params_delta = [gamma, k, complex(delta, h), r]
    Y_delta = solve_sir_complex(params_delta)
    S_delta = np.imag(Y_delta[0]) / h
    I_delta = np.imag(Y_delta[1]) / h
    R_delta = np.imag(Y_delta[2]) / h
    
    # Sensitivity w.r.t r
    params_r = [gamma, k, delta, complex(r, h)]
    Y_r = solve_sir_complex(params_r)
    S_r = np.imag(Y_r[0]) / h
    I_r = np.imag(Y_r[1]) / h
    R_r = np.imag(Y_r[2]) / h
    
    # Create plots comparing sensitivity equations vs complex-step
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    param_names = [r'\gamma', 'k', r'\delta', 'r']
    state_names = ['S', 'I', 'R']
    
    sens_eq_data = [
        [S_gamma_sen, I_gamma_sen, R_gamma_sen],
        [S_k_sen, I_k_sen, R_k_sen],
        [S_delta_sen, I_delta_sen, R_delta_sen],
        [S_r_sen, I_r_sen, R_r_sen]
    ]
    
    complex_data = [
        [S_gamma, I_gamma, R_gamma],
        [S_k, I_k, R_k],
        [S_delta, I_delta, R_delta],
        [S_r, I_r, R_r]
    ]
    
    for i, param in enumerate(param_names):
        for j, state in enumerate(state_names):
            ax = axes[i, j]
            ax.plot(t_vals, sens_eq_data[i][j], '-b', linewidth=2, label='Sensitivity Eq')
            ax.plot(t_vals, complex_data[i][j], '--r', linewidth=2, label='Complex-Step')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'${state}_{{{param}}}$')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Figures/SIR_sensitivities.png', dpi=150)
    plt.show()
    
    # Parameter subset selection
    sens_mat = np.column_stack([R_gamma, R_k, R_delta, R_r])
    eta = 1e-10
    identifiable, unidentifiable = pss_eig(sens_mat, eta)
    
    print("\n=== Parameter Subset Selection Results ===")
    param_list = ['gamma', 'k', 'delta', 'r']
    print(f"Identifiable parameters: {[param_list[i] for i in identifiable]}")
    print(f"Unidentifiable parameters: {[param_list[i] for i in unidentifiable]}")


if __name__ == "__main__":
    main()
