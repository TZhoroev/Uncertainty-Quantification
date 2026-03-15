# Project 1: Sensitivity Analysis

## Introduction

Sensitivity analysis is a fundamental tool in uncertainty quantification that helps us understand how changes in model parameters affect model outputs. This project explores both local and global sensitivity analysis methods applied to physical and biological systems.

Local sensitivity analysis examines the effect of small perturbations around a nominal parameter value. Global sensitivity analysis, in contrast, explores the entire parameter space to understand how uncertainty in inputs propagates to outputs.

## Problem 1: Spring Model Sensitivities

### Background

We consider a simple spring model where the displacement depends on physical parameters. The goal is to compute the sensitivity of the displacement with respect to model parameters using three approaches:

1. Finite difference approximation
2. Complex step derivative approximation  
3. Analytical derivatives

### Mathematical Formulation

For a function f(q), the finite difference approximation of the derivative is:

    df/dq ≈ [f(q + h) - f(q)] / h

The complex step method uses:

    df/dq ≈ Im[f(q + ih)] / h

where i is the imaginary unit. This method achieves machine precision accuracy without subtractive cancellation errors.

### Results

The figure below shows the comparison of all three methods:

![Spring Model Sensitivities](Figures/8_5_1.jpg)

The complex step method matches the analytical solution to machine precision, while finite differences show truncation errors for small step sizes and round-off errors for very small step sizes.

![Convergence Analysis](Figures/8_5_2.jpg)

## Problem 2: SIR Model Sensitivities and Identifiability

### Background

The SIR (Susceptible-Infected-Recovered) model describes the spread of infectious diseases. The model equations are:

    dS/dt = delta*(N - S) - gamma*I*S
    dI/dt = gamma*I*S - (r + delta)*I  
    dR/dt = r*I - delta*R

where:
- S, I, R are susceptible, infected, and recovered populations
- gamma is the infection rate
- delta is the birth/death rate
- r is the recovery rate
- N is the total population

### Sensitivity Analysis

We compute sensitivities of the infected population with respect to each parameter using the complex step method. The sensitivity equations are solved alongside the state equations.

### Parameter Identifiability

The Fisher Information Matrix (FIM) is computed as:

    F = (1/sigma^2) * X^T * X

where X is the sensitivity matrix. The covariance matrix of the estimated parameters is approximately:

    V ≈ sigma^2 * (X^T * X)^(-1)

Parameters are identifiable if the FIM is well-conditioned (all eigenvalues are bounded away from zero).

### Results

![SIR Sensitivities](Figures/8_8_1.jpg)

The sensitivity plots show how each parameter influences the infected population over time.

![Parameter Correlations](Figures/8_8_2.jpg)

![Identifiability Analysis](Figures/8_8_3.jpg)

![Fisher Information](Figures/8_8_4.jpg)

## Problem 3: Heat Equation Parameter Identifiability

### Background

We analyze a heat conduction problem in a rod with convective boundary conditions. The steady-state temperature distribution depends on thermal conductivity and convection coefficients.

### Governing Equations

The steady-state heat equation is:

    d/dx(k * dT/dx) = 0

with boundary conditions involving convective heat transfer.

### Identifiability Analysis

We compute the sensitivity of temperature with respect to model parameters and construct the Fisher Information Matrix to assess parameter identifiability.

### Results

![Heat Equation Sensitivities](Figures/8_9_1.jpg)

![Parameter Correlations](Figures/8_9_2.jpg)

![Eigenvalue Analysis](Figures/8_9_3.jpg)

## Problem 4: Global Sensitivity Analysis

### Background

Global sensitivity analysis methods explore the entire parameter space rather than just local perturbations. We apply two methods:

1. Morris screening (elementary effects)
2. Sobol indices (variance-based)

### Morris Screening

The Morris method computes elementary effects by taking one-at-a-time perturbations at random points in the parameter space. For each parameter, we compute:

- Mean of absolute elementary effects (importance)
- Standard deviation of elementary effects (interaction/nonlinearity)

### Sobol Indices

Sobol indices decompose the output variance into contributions from individual parameters and their interactions:

- First-order index Si: fraction of variance due to parameter i alone
- Total-effect index STi: fraction of variance due to parameter i including interactions

### Results

![Morris Screening Results](Figures/9_6_1.jpg)

Parameters with high mean absolute effect are important. Parameters with high standard deviation show strong interactions or nonlinear effects.

![Sobol Indices](Figures/9_6_2.jpg)

The Sobol indices quantify the relative importance of each parameter in explaining output variance.

## Summary

This project demonstrated several key concepts in sensitivity analysis:

1. The complex step method provides machine-precision derivatives without the truncation-roundoff tradeoff of finite differences.

2. Local sensitivity analysis reveals how parameters influence model outputs near nominal values.

3. The Fisher Information Matrix connects sensitivities to parameter identifiability and estimation uncertainty.

4. Global methods like Morris screening and Sobol indices provide complementary information about parameter importance across the entire feasible range.

## Code Files

| File | Description |
|------|-------------|
| UQ_8_5.m / UQ_8_5.py | Spring model sensitivities |
| UQ_8_8.m / UQ_8_8.py | SIR model analysis |
| UQ_8_9.m / UQ_8_9.py | Heat equation identifiability |
| UQ_9_6.m / UQ_9_6.py | Global sensitivity analysis |

## References

1. Smith, R.C. (2013). Uncertainty Quantification: Theory, Implementation, and Applications. SIAM.
2. Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer. Wiley.
