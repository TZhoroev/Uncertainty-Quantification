# Project 4: Uncertainty Propagation

## Introduction

Once we have characterized uncertainty in model parameters, the next step is propagating that uncertainty through the model to quantify uncertainty in predictions. This project explores both frequentist and Bayesian approaches to constructing intervals for model predictions.

## Types of Intervals

### Confidence Intervals (Frequentist)

A 95% confidence interval means: if we repeated the experiment many times, 95% of the computed intervals would contain the true parameter value.

For linear regression, the confidence interval for the mean response at x is:

    y_hat +/- t_{alpha/2, n-p} * s * sqrt(x^T (X^T X)^(-1) x)

### Prediction Intervals (Frequentist)

Prediction intervals account for both parameter uncertainty and measurement noise:

    y_hat +/- t_{alpha/2, n-p} * s * sqrt(1 + x^T (X^T X)^(-1) x)

### Credible Intervals (Bayesian)

A 95% credible interval contains the true parameter value with 95% probability given the observed data. This is computed from the posterior distribution.

## Problem 1: Height-Weight Model Intervals

### Background

We fit a linear regression model relating height and weight, then construct confidence and prediction intervals.

### Model

    weight = beta_0 + beta_1 * height + epsilon

### Results

![Height-Weight Regression](Figures & Data/1_1.eps)

The confidence band (narrower) represents uncertainty in the mean relationship, while the prediction band (wider) includes the expected variation in individual observations.

![Confidence and Prediction Intervals](Figures & Data/1_2.eps)

## Problem 2: Aluminum Rod Bayesian Intervals

### Background

We revisit the heat conduction problem using Bayesian inference to construct credible intervals and predictive intervals.

### MCMC for Posterior Sampling

We sample from the posterior distribution of (Q, h, sigma^2) using DRAM MCMC.

### Posterior Predictive Distribution

For each posterior sample theta^(j), we compute the model prediction:

    y^(j)(x) = f(x; theta^(j))

The collection of predictions forms the posterior predictive distribution.

### Results

![MCMC Chains for Q and h](Figures & Data/1_1.eps)

![Posterior Densities](Figures & Data/1_2.eps)

![Bayesian Prediction Bands](Figures & Data/4.eps)

The shaded region shows the 95% credible interval for predictions. The data points fall within this band, indicating good model fit.

## Problem 3: SIR Model Credible Intervals

### Background

We propagate posterior uncertainty in SIR parameters through the model to obtain credible intervals for the infected population over time.

### Procedure

1. Run MCMC to sample from p(gamma, delta, r | data)
2. For each posterior sample, solve the SIR equations
3. Compute pointwise quantiles of the solution trajectories

### Results

![SIR Credible Bands](Figures & Data/3_d1.jpg)

The credible band captures the uncertainty in the epidemic trajectory due to parameter uncertainty.

![SIR Model Components](Figures & Data/3_d2.jpg)

## Problem 4: Frequentist vs Bayesian Comparison

### Philosophical Differences

**Frequentist interpretation**: The parameter is fixed but unknown. Probability statements are about the procedure, not the parameter.

**Bayesian interpretation**: The parameter is treated as a random variable. Probability statements directly describe our beliefs about the parameter.

### Practical Differences

In many cases, especially with sufficient data and weak priors, frequentist confidence intervals and Bayesian credible intervals give similar numerical results.

Key differences arise when:
- Sample sizes are small
- Prior information is substantial
- The likelihood is not well-approximated by a normal distribution

### Results

![Interval Comparison](Figures & Data/5.eps)

For the heat conduction model with adequate data, both approaches give similar intervals. The Bayesian approach additionally provides a coherent framework for incorporating prior knowledge and making probabilistic predictions.

## Summary

Key concepts covered in this project:

1. **Confidence intervals** quantify sampling variability in frequentist estimation.

2. **Prediction intervals** add observation noise to account for future measurements.

3. **Credible intervals** represent posterior probability for parameters.

4. **Posterior predictive intervals** propagate full posterior uncertainty through the model.

5. Both approaches often agree numerically, but differ in interpretation.

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Frequentist intervals |
| Problem2.m / Problem2.py | Bayesian intervals for heat model |
| Problem3a.m / Problem3.py | SIR credible intervals |
| Problem4.m / Problem4.py | Frequentist vs Bayesian comparison |

## References

1. Wasserman, L. (2004). All of Statistics. Springer.
2. Gelman, A. et al. (2013). Bayesian Data Analysis. CRC Press.
