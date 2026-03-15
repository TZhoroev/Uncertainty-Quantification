# Project 6: Model Discrepancy

## Introduction

In practice, computational models never perfectly represent physical reality. Model discrepancy (also called model inadequacy or structural uncertainty) refers to the systematic difference between a model's predictions and the true physical process, even when parameters are perfectly known.

This project explores model discrepancy using the Dittus-Boelter correlation for heat transfer.

## The Dittus-Boelter Equation

### Background

The Dittus-Boelter equation is an empirical correlation for turbulent flow heat transfer:

    Nu = theta_1 * Re^{theta_2} * Pr^{theta_3}

where:
- Nu is the Nusselt number (dimensionless heat transfer coefficient)
- Re is the Reynolds number (ratio of inertial to viscous forces)
- Pr is the Prandtl number (ratio of momentum to thermal diffusivity)
- theta_1, theta_2, theta_3 are empirical coefficients

The classical values are theta_1 = 0.023, theta_2 = 0.8, theta_3 = 0.4.

### Physical Interpretation

The Nusselt number represents the enhancement of heat transfer due to convection compared to pure conduction. The correlation captures the experimental observation that heat transfer increases with flow rate (Re) and depends on the fluid properties (Pr).

## Problem: Analyzing Model Discrepancy

### Data

We have experimental measurements of Nu at various (Re, Pr) conditions. The data exhibits scatter around the Dittus-Boelter predictions.

### Sources of Discrepancy

1. **Measurement error**: Random noise in the experimental data
2. **Model form error**: The power-law form may not exactly represent the physics
3. **Parameter uncertainty**: The "optimal" parameters may vary between experiments

### Analysis Approach

We perform:
1. Nonlinear least squares to estimate the parameters
2. Residual analysis to identify systematic patterns
3. MCMC to quantify parameter uncertainty
4. Assessment of model adequacy

## Parameter Estimation

### Initial Estimates

Starting from the classical values, we optimize using nonlinear least squares:

    min_{theta} sum_i [Nu_data(i) - Nu_model(Re_i, Pr_i; theta)]^2

### Sensitivity Matrix

The Jacobian matrix contains partial derivatives:

    X_ij = partial Nu_i / partial theta_j

For the Dittus-Boelter model:
- partial Nu / partial theta_1 = Re^{theta_2} * Pr^{theta_3}
- partial Nu / partial theta_2 = theta_1 * Re^{theta_2} * Pr^{theta_3} * ln(Re)
- partial Nu / partial theta_3 = theta_1 * Re^{theta_2} * Pr^{theta_3} * ln(Pr)

### Parameter Identifiability

We compute the Fisher Information Matrix and check for ill-conditioning using SVD:

    F = X^T X

Parameters are identifiable if F is well-conditioned.

## Residual Analysis

### Checking Model Adequacy

We plot residuals (data - model predictions) against predicted values:

![Residual Plot Initial](Figures&Data/11.eps)

A good model should have residuals that:
- Are randomly scattered around zero
- Show constant variance (homoscedasticity)
- Have no systematic patterns

### Results

![Residual Plot After Estimation](Figures&Data/12.eps)

The residuals fall within the 2-sigma bounds, indicating reasonable model fit.

## Bayesian Analysis

### MCMC Implementation

We sample from the posterior distribution:

    p(theta, sigma^2 | data) proportional to p(data | theta, sigma^2) * p(theta) * p(sigma^2)

Using DRAM with 50,000 samples (10,000 burn-in).

### Results

![Parameter Chains](Figures&Data/4.eps)

The chains show good mixing after burn-in.

![Parameter Densities](Figures&Data/5.eps)

![Pairwise Scatter](Figures&Data/6.eps)

The posterior distributions reveal:
- Moderate uncertainty in all parameters
- Strong correlation between theta_1 and theta_2
- The data are informative about all parameters

![Variance Chain](Figures&Data/7.eps)

## Model Discrepancy Formulation

### Statistical Model

A more complete model acknowledges discrepancy:

    y = f(x; theta) + delta(x) + epsilon

where:
- f(x; theta) is the computational model
- delta(x) is the model discrepancy (systematic)
- epsilon is measurement error (random)

### Challenges

Separating model discrepancy from parameter uncertainty is difficult without additional information. Common approaches include:
- Using multiple data sources
- Physical constraints on the discrepancy
- Hierarchical models

## Summary

Key findings from this project:

1. The Dittus-Boelter model provides reasonable predictions, but residuals suggest some systematic model error.

2. All three parameters are identifiable from the data.

3. MCMC reveals correlations between parameters that would be missed by point estimates.

4. Model discrepancy is an important source of uncertainty that complements parameter uncertainty.

## Code Files

| File | Description |
|------|-------------|
| Final.m / Final.py | Complete analysis |
| PSS_SVD.m | Parameter subset selection using SVD |

## References

1. Kennedy, M.C. and O'Hagan, A. (2001). Bayesian calibration of computer models. JRSS B.
2. Smith, R.C. (2013). Uncertainty Quantification. SIAM.
