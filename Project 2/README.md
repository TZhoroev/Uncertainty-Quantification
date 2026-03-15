# Project 2: Parameter Estimation

## Introduction

Parameter estimation is the process of finding the best values for model parameters given observed data. This project explores classical estimation techniques including ordinary least squares (OLS) and constrained optimization for nonlinear models.

Understanding parameter estimation is essential for model calibration and forms the foundation for both frequentist and Bayesian uncertainty quantification.

## Problem 1: Heat Model Parameter Estimation

### Background

We consider heat conduction in a copper rod with convective heat loss. Temperature measurements are taken along the rod, and we wish to estimate the heat flux Q and convection coefficient h.

### Physical Model

The steady-state temperature distribution in the rod satisfies:

    d^2u/dx^2 - gamma^2 * (u - u_amb) = 0

where gamma^2 = 2(a+b)h / (abk), with a and b being the cross-sectional dimensions, k the thermal conductivity, and h the convection coefficient.

The analytical solution is:

    u(x) = c1*exp(-gamma*x) + c2*exp(gamma*x) + u_amb

where c1 and c2 depend on the boundary conditions involving the heat flux Q.

### Parameter Estimation

We minimize the sum of squared residuals:

    J(Q, h) = sum_i [u_data(x_i) - u_model(x_i; Q, h)]^2

using constrained optimization to ensure physically meaningful parameter values (h > 0).

### Results

The estimated parameters provide an excellent fit to the temperature data.

![Heat Model Fit](Figures& Data/1a.eps)

The sensitivity analysis shows that both Q and h are identifiable from the data, with some correlation between the estimates.

![Parameter Correlation](Figures& Data/1b.eps)

## Problem 2: Helmholtz Energy Model

### Background

The Helmholtz free energy is a thermodynamic quantity that can be modeled as a polynomial in pressure for certain materials:

    psi(p) = alpha1*p^2 + alpha11*p^4 + alpha111*p^6

We estimate the coefficients from experimental data using ordinary least squares.

### Estimation Procedure

Since the model is linear in the parameters, we can write:

    psi = X * theta + epsilon

where X is the design matrix with columns [p^2, p^4, p^6] and theta = [alpha1, alpha11, alpha111]^T.

The OLS estimator is:

    theta_hat = (X^T X)^(-1) X^T psi_data

The covariance of the estimator is:

    V = sigma^2 * (X^T X)^(-1)

where sigma^2 is estimated from the residuals.

### Results

![Helmholtz Model Fit](Figures& Data/2a.eps)

The polynomial model captures the nonlinear relationship between pressure and Helmholtz energy.

![Residual Analysis](Figures& Data/2b.eps)

The residuals show no systematic pattern, indicating a good model fit.

![Parameter Distributions](Figures& Data/2c.eps)

## Problem 3: SIR Model Parameter Distributions

### Background

For the SIR epidemic model, we estimate the parameters gamma (infection rate), delta (birth/death rate), and r (recovery rate) from time series data of infected individuals.

### Nonlinear Estimation

Since the SIR model is nonlinear in its parameters, we use iterative optimization. The sensitivity matrix is computed using the complex-step derivative:

    dI/d(gamma) ≈ Im[I(gamma + i*h)] / h

### Asymptotic Distribution

Under regularity conditions, the OLS estimator for nonlinear models is asymptotically normal:

    theta_hat ~ N(theta_true, V)

where V = sigma^2 * (X^T X)^(-1) and X is the sensitivity matrix evaluated at the estimated parameters.

### Results

![SIR Model Fit](Figures& Data/3a.eps)

The fitted model captures the epidemic dynamics well.

![Parameter Marginal Distributions](Figures& Data/3b.eps)

The marginal distributions show the uncertainty in each parameter estimate. The parameters are correlated, as seen in the joint distributions.

## Summary

This project covered key concepts in parameter estimation:

1. Nonlinear least squares estimation using iterative optimization methods.

2. Sensitivity analysis to compute the Jacobian matrix for nonlinear models.

3. Construction of confidence regions using the asymptotic covariance matrix.

4. The importance of checking residuals for model adequacy.

## Key Equations

For OLS with linear models:
- Estimator: theta_hat = (X^T X)^(-1) X^T y
- Covariance: V = sigma^2 (X^T X)^(-1)
- sigma^2 estimate: s^2 = RSS / (n - p)

For nonlinear models, replace X with the sensitivity matrix evaluated at theta_hat.

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Heat model estimation |
| Problem2.m / Problem2.py | Helmholtz energy OLS |
| Problem3.m / Problem3.py | SIR model estimation |

## References

1. Seber, G.A.F. and Wild, C.J. (2003). Nonlinear Regression. Wiley.
2. Banks, H.T. and Tran, H.T. (2009). Mathematical and Experimental Modeling of Physical and Biological Processes. CRC Press.
