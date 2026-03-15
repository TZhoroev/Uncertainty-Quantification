# Project 5: Surrogate Models

## Introduction

Complex computational models can be expensive to evaluate, making tasks like uncertainty propagation and optimization computationally prohibitive. Surrogate models (also called metamodels or emulators) provide fast approximations to expensive simulations.

This project explores polynomial surrogates and Gaussian process regression as surrogate modeling techniques.

## Problem 1: Polynomial Surrogate with Latin Hypercube Sampling

### Background

We build a polynomial surrogate model using Latin Hypercube Sampling (LHS) to efficiently sample the parameter space.

### Latin Hypercube Sampling

LHS creates a space-filling design by:
1. Dividing each parameter range into n equal intervals
2. Sampling once from each interval
3. Randomly pairing the samples across dimensions

This ensures better coverage than random sampling while maintaining the marginal distributions.

### Polynomial Surrogate

We approximate the expensive function f(x) with a polynomial:

    f_hat(x) = sum_j c_j * phi_j(x)

where phi_j are polynomial basis functions (e.g., monomials, Legendre polynomials).

The coefficients are found by least squares:

    c = (Phi^T Phi)^(-1) Phi^T y

where Phi_ij = phi_j(x_i) is the design matrix.

### Results

![LHS Sample Points](Figures&Data/11.eps)

The Latin Hypercube design provides good coverage of the parameter space with fewer points than a full grid.

![Surrogate Fit](Figures&Data/12.eps)

The polynomial surrogate captures the main features of the true function.

## Problem 2: Legendre Polynomial Surrogate

### Background

Legendre polynomials form an orthogonal basis on [-1, 1], which provides numerical stability for regression.

### Legendre Polynomials

The first few Legendre polynomials are:
- P_0(x) = 1
- P_1(x) = x
- P_2(x) = (3x^2 - 1)/2
- P_3(x) = (5x^3 - 3x)/2

They satisfy the orthogonality relation:

    integral_{-1}^{1} P_m(x) P_n(x) dx = 2/(2n+1) * delta_{mn}

### Surrogate Construction

For multi-dimensional problems, we use tensor products of 1D Legendre polynomials or total-degree truncation to control the number of terms.

### Results

![Legendre Basis Functions](Figures&Data/4.eps)

![Legendre Surrogate Error](Figures&Data/5.eps)

The Legendre basis provides a stable and accurate surrogate representation.

## Problem 3: Gaussian Process Regression

### Background

Gaussian Process (GP) regression is a non-parametric approach that treats the unknown function as a realization of a Gaussian process.

### GP Prior

A Gaussian process is fully specified by:
- Mean function: m(x) = E[f(x)]
- Covariance function: k(x, x') = Cov[f(x), f(x')]

Common covariance functions include:

**Squared Exponential (RBF):**
    k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))

**Matern 5/2:**
    k(x, x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)

where r = ||x - x'||.

### GP Posterior

Given training data (X, y), the posterior predictive distribution at test point x* is:

    f(x*) | X, y ~ N(mu*, sigma*^2)

where:
    mu* = k(x*, X) [K + sigma_n^2 I]^(-1) y
    sigma*^2 = k(x*, x*) - k(x*, X) [K + sigma_n^2 I]^(-1) k(X, x*)

and K_ij = k(x_i, x_j).

### Results

![GP Fit with Uncertainty](Figures&Data/6.eps)

The GP provides both a mean prediction and uncertainty bands. The uncertainty increases away from training points.

![GP Hyperparameter Effects](Figures&Data/7.eps)

The length scale l controls the smoothness, and sigma_f controls the amplitude of variations.

## Comparison of Methods

| Method | Pros | Cons |
|--------|------|------|
| Polynomial | Simple, fast evaluation | May need many terms for complex functions |
| Legendre | Numerically stable | Limited flexibility |
| Gaussian Process | Uncertainty quantification, flexible | Scales poorly with data size |

## Summary

Key concepts covered:

1. **Latin Hypercube Sampling** provides efficient space-filling designs.

2. **Polynomial surrogates** are simple and fast but may require careful basis selection.

3. **Gaussian processes** provide principled uncertainty quantification and adapt to data automatically.

4. The choice of surrogate depends on the problem: dimensionality, smoothness, and whether uncertainty estimates are needed.

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Polynomial surrogate with LHS |
| Problem2.m / Problem2.py | Legendre polynomial surrogate |
| Problem3.m / Problem3.py | Gaussian process regression |

## References

1. Rasmussen, C.E. and Williams, C.K.I. (2006). Gaussian Processes for Machine Learning. MIT Press.
2. Xiu, D. (2010). Numerical Methods for Stochastic Computations. Princeton University Press.
