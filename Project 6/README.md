# Project 6: Model Discrepancy

**Math 540: Uncertainty Quantification**

**Author:** Tilekbek Zhoroev

---

## Problem: Dittus-Boelter Equation Analysis

### Problem Statement

Consider the Dittus-Boelter equation:

$$Nu = \theta_1 Re^{\theta_2} Pr^{\theta_3}$$

where $Nu$, $Re$, and $Pr$ respectively denote the Nusselt, Reynolds, and Prandtl numbers.

Reported nominal parameter values are:

$$\theta_0 = [0.023, 0.8, 0.4]$$

Data is provided in db_data.txt where Re = db_data(:,1); Pr = db_data(:,2); Nu = db_data(:,3).

Construct the Fisher information matrix and discuss the identifiability of the parameters. Use DRAM to compute posterior densities for the parameters.

### Solution

#### Initial Residual Analysis

Using the nominal parameter values, we examine the residuals. From Figure 1(a), we observe that the residuals are not identically and independently distributed, indicating that the nominal parameters may not be optimal for this dataset.

#### Optimal Parameter Estimation

Using MATLAB's `lsqnonlin.m`, we obtain optimal parameter values:

$$\theta = [0.004, 0.986, 0.411]$$

#### Variance Estimation

With $n = 56$ data points and $p = 3$ parameters, the error variance estimate is:

$$\sigma^2 = \frac{1}{n-p} \mathbf{R}^\top \mathbf{R} = 162.1507$$

where $\mathbf{R}$ is the vector of residuals.

The residual plot using optimal parameters (Figure 1b) shows identically and independently distributed residuals.

![Initial Residuals](Figures&Data/11.eps)

![Optimal Residuals](Figures&Data/12.eps)

**Figure 1:** (a) Residuals with given nominal parameter values; (b) Residuals with optimal parameter values.

#### Sensitivity Matrix

The sensitivity matrix is:

$$X = \begin{bmatrix} Re_1^{\theta_2} Pr_1^{\theta_3} & \theta_1 Re_1^{\theta_2} Pr_1^{\theta_3} \ln(Re_1) & \theta_1 Re_1^{\theta_2} Pr_1^{\theta_3} \ln(Pr_1) \\ Re_2^{\theta_2} Pr_2^{\theta_3} & \theta_1 Re_2^{\theta_2} Pr_2^{\theta_3} \ln(Re_2) & \theta_1 Re_2^{\theta_2} Pr_2^{\theta_3} \ln(Pr_2) \\ \vdots & \vdots & \vdots \\ Re_n^{\theta_2} Pr_n^{\theta_3} & \theta_1 Re_n^{\theta_2} Pr_n^{\theta_3} \ln(Re_n) & \theta_1 Re_n^{\theta_2} Pr_n^{\theta_3} \ln(Pr_n) \end{bmatrix}_{n \times p}$$

#### Fisher Information Matrix

The Fisher information matrix is:

$$F = X^\top X = \begin{bmatrix} 9.3892 \times 10^{10} & 3.5764 \times 10^{9} & 1.3305 \times 10^{9} \\ 3.5764 \times 10^{9} & 1.3700 \times 10^{8} & 4.9713 \times 10^{7} \\ 1.3305 \times 10^{9} & 4.9713 \times 10^{7} & 2.1098 \times 10^{7} \end{bmatrix}$$

#### Eigenvalue Analysis

The eigenvalues of the Fisher matrix are:

$$\lambda = [2.9325 \times 10^{5}, \quad 2.7239 \times 10^{6}, \quad 9.4047 \times 10^{10}]$$

Because all eigenvalues are sufficiently large (bounded away from zero), we conclude that **all parameters are identifiable**.

#### Covariance Matrix

The covariance matrix is:

$$V = \sigma^2 (X^\top X)^{-1} = \begin{bmatrix} 9.0474 \times 10^{-7} & -2.0102 \times 10^{-5} & -9.6884 \times 10^{-6} \\ -2.0102 \times 10^{-5} & 4.5482 \times 10^{-4} & 1.9604 \times 10^{-4} \\ -9.6884 \times 10^{-6} & 1.9604 \times 10^{-4} & 1.5674 \times 10^{-4} \end{bmatrix}$$

### Bayesian Analysis with DRAM

We use the optimal parameter values as initial values and the Frequentist covariance matrix as the initial proposal covariance.

#### Convergence

Figure 2 shows the chain plots for each parameter and pairwise sample plots. The MCMC algorithm has converged. The pairwise plots demonstrate parameter correlations, and all parameters are identifiable, consistent with the Frequentist analysis.

![Chain Theta1](Figures&Data/4.eps)

![Chain Theta2](Figures&Data/5.eps)

![Chain Theta3](Figures&Data/6.eps)

**Figure 2:** Chain plots and pairwise sample plots.

#### Bayesian Parameter Estimates

$$\theta = [0.004, 0.982, 0.409]$$

#### Bayesian Variance Estimate

$$\sigma^2 = 168.5046$$

#### Marginal Distributions

The marginal distributions of each parameter are shown in Figure 3.

![Marginal Distributions](Figures&Data/7.eps)

**Figure 3:** Marginal distributions of (a) $\theta_1$; (b) $\theta_2$; (c) $\theta_3$.

### Comparison of Frequentist and Bayesian Results

| Parameter | Frequentist | Bayesian |
|-----------|-------------|----------|
| $\theta_1$ | 0.004 | 0.004 |
| $\theta_2$ | 0.986 | 0.982 |
| $\theta_3$ | 0.411 | 0.409 |
| $\sigma^2$ | 162.15 | 168.50 |

The Bayesian and Frequentist estimates are in close agreement, validating both approaches.

### Conclusions

1. The nominal Dittus-Boelter parameters $[0.023, 0.8, 0.4]$ do not fit this dataset well.

2. Optimized parameters $[\sim 0.004, \sim 0.98, \sim 0.41]$ provide better fit.

3. All three parameters are identifiable based on the Fisher information analysis.

4. DRAM posterior distributions confirm parameter identifiability through pairwise plots.

5. There is good agreement between Frequentist and Bayesian parameter estimates.

---

## Code Files

| File | Description |
|------|-------------|
| Final.m / Final.py | Complete Dittus-Boelter analysis |
| PSS_SVD.m | Parameter subset selection using SVD |

## References

1. Kennedy, M.C. and O'Hagan, A. (2001). Bayesian calibration of computer models. *JRSS B*.
2. Dittus, F.W. and Boelter, L.M.K. (1930). Heat transfer in automobile radiators of the tubular type. *University of California Publications in Engineering*.
3. Smith, R.C. (2013). *Uncertainty Quantification*. SIAM.
