# Project 2: Parameter Estimation

**Math 540: Uncertainty Quantification**

**Author:** Tilekbek Zhoroev

---

## Problem 1: Heat Model Parameter Estimation (Copper Rod)

### Problem Statement

Repeat the analysis of Example 11.22 for the steady state heat model using the copper data in Table 1 and the values $k = 4.01$ for the thermal conductivity and $u_{amb} = 22.28$. Do your residuals appear to be iid? Use a Q-Q plot to establish whether the residuals are normally distributed.

| x | 10 | 14 | 18 | 22 | 26 | 30 | 34 | 38 | 42 | 46 | 50 | 54 | 58 | 62 | 66 |
|---|----|----|----|----|----|----|----|----|----|----|----|----|----|----|---|
| Temp | 66.04 | 60.04 | 54.81 | 50.42 | 46.74 | 43.66 | 40.76 | 38.49 | 36.42 | 34.77 | 33.18 | 32.36 | 31.56 | 30.91 | 30.56 |

**Table 1:** Steady-state temperatures measured at locations $x$ for a copper rod.

### Solution

We are given the temperature of the copper rod at equidistant points. Using the analytic solution of the heat equation and given data, we constructed a least squares problem. Using `fminsearch.m`, we obtained parameter estimates:

$$\Phi = -9.9265, \quad h = 0.0014$$

### Model Fitting

The graph of our model using estimated parameters and data shows good fitting. However, we observe from the residual plot that residuals are identical but not independent. The Q-Q plot analysis shows that residuals do not follow a straight line, so residuals are not from a normal distribution.

### Variance and Covariance Estimation

With $n = 16$ data points and $p = 2$ parameters, we compute the error variance estimate:

$$\sigma^2 = \frac{1}{n-p} \mathbf{R}^\top \mathbf{R} = 0.0529$$

where $\mathbf{R}$ is the vector of residuals.

Using the sensitivity matrix $X$, we obtain the covariance matrix:

$$V = \sigma^2 [X^\top X]^{-1} = \begin{bmatrix} 0.0090 & -1.2117 \times 10^{-6} \\ -1.2117 \times 10^{-6} & 1.7726 \times 10^{-10} \end{bmatrix}$$

### Standard Deviations

The estimated standard deviations are:

$$\sigma = 0.2301, \quad \sigma_\Phi = 0.0947, \quad \sigma_h = 1.3314 \times 10^{-5}$$

### 95% Confidence Intervals

Using the t-distribution:

$$\Phi \in [-10.1310, -9.7219]$$

$$h \in [0.0014, 0.0015]$$

![Sensitivity Phi](Figures& Data/1a.eps)

![Sensitivity h](Figures& Data/1b.eps)

**Figure 1:** Sensitivity equations: (a) $\frac{\partial u}{\partial \Phi}$; (b) $\frac{\partial u}{\partial h}$.

![Model Fit](Figures& Data/2a.eps)

![Residuals](Figures& Data/2b.eps)

![Q-Q Plot](Figures& Data/2c.eps)

**Figure 2:** (a) The graph of the data and model for the uninsulated copper rod; (b) The residual plot; (c) Q-Q plot of the residuals.

---

## Problem 2: Helmholtz Energy Model (OLS Estimation)

### Problem Statement

Consider the Helmholtz energy:

$$\psi(P, \theta) = \alpha_1 P^2 + \alpha_{11} P^4 + \alpha_{111} P^6$$

where $P$ is the polarization on the interval $[0, 0.8]$ and $\theta = [\alpha_1, \alpha_{11}, \alpha_{111}]^\top$ are parameters with nominal values $\alpha_1 = -389.4$, $\alpha_{11} = 761.3$, $\alpha_{111} = 61.5$.

### Part (a): Variance Estimation for Different Sample Sizes

For $n = 81, 161, 801$ equally spaced polarization values $P_i = (i-1)\Delta P$, $\Delta P = \frac{0.8}{n-1}$, we compute the model response and observations:

$$Y_i = \psi(P_i, \theta) + \varepsilon_i$$

where $\varepsilon_i \sim_{iid} N(0, \sigma^2)$ with $\sigma = 2.2$.

The OLS estimate for observation variance is:

$$\hat{\sigma}^2 = \frac{1}{n-p} \mathbf{R}^\top \mathbf{R}$$

**Results:**

| $n$ | $\hat{\sigma}$ |
|-----|----------------|
| 81 | 2.3841 |
| 161 | 2.2594 |
| 801 | 2.1791 |

As data points increase, the estimated variance converges to the true variance $\sigma = 2.2$.

### Part (b): Normal Equations and Parameter Estimation

Using the observation model:

$$Y_i = \psi(P_i, \theta) + \varepsilon_i = [P_i^2, P_i^4, P_i^6] \begin{bmatrix} \alpha_1 \\ \alpha_{11} \\ \alpha_{111} \end{bmatrix} + \varepsilon_i$$

The design matrix is:

$$X = \begin{bmatrix} P_1^2 & P_1^4 & P_1^6 \\ P_2^2 & P_2^4 & P_2^6 \\ \vdots & \vdots & \vdots \\ P_n^2 & P_n^4 & P_n^6 \end{bmatrix}$$

The normal equations give the estimated parameters:

$$(X^\top X)\hat{\theta} = X^\top Y \implies \hat{\theta} = (X^\top X)^{-1} X^\top Y$$

$$\hat{\theta} = \begin{bmatrix} \hat{\alpha}_1 \\ \hat{\alpha}_{11} \\ \hat{\alpha}_{111} \end{bmatrix} = \begin{bmatrix} -390.0709 \\ 764.5939 \\ 57.2555 \end{bmatrix}$$

### Covariance Matrix

$$V = \begin{bmatrix} 22.5543 & -110.0835 & 123.4682 \\ -110.0835 & 585.0489 & -690.2614 \\ 123.4682 & -690.2614 & 842.2265 \end{bmatrix}$$

### Correlation Coefficients

The correlation coefficient between two random variables $X$ and $Y$ is:

$$\rho_{X,Y} = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}$$

Computing correlations:

$$\rho_{\alpha_1, \alpha_{11}} = \frac{-110.0835}{\sqrt{22.543 \times 585.0489}} = -0.9585$$

$$\rho_{\alpha_1, \alpha_{111}} = \frac{123.4681}{\sqrt{22.543 \times 842.2265}} = 0.8960$$

$$\rho_{\alpha_{11}, \alpha_{111}} = \frac{-690.2614}{\sqrt{585.0489 \times 842.2265}} = -0.9833$$

All parameters are mutually correlated with strong correlation. Thus, we cannot assume these parameters are mutually independent when employing global sensitivity analysis.

![Residual Plot](Figures& Data/3a.eps)

![Model and Observations](Figures& Data/3b.eps)

**Figure 3:** (a) Residual plot with $2\sigma$ intervals; (b) Graph of the model and observations.

---

## Problem 3: SIR Model Parameter Estimation

### Problem Statement

Consider the SIR model:

$$\frac{dS}{dt} = \delta N - \delta S - \gamma IS, \quad S(0) = 900$$

$$\frac{dI}{dt} = \gamma IS - (r + \delta)I, \quad I(0) = 100$$

$$\frac{dR}{dt} = rI - \delta R, \quad R(0) = 0$$

where $\gamma, r, \delta \in [0, 1]$.

### Part (a): Parameter Estimation

Using `fminsearch.m` with the data from SIR.txt, we obtained:

$$\gamma = 0.0100, \quad \delta = 0.1953, \quad r = 0.7970$$

The residual plot shows that errors are identically and independently distributed. The fitted model shows good agreement with observational data.

### Part (b): Sensitivity Analysis and Covariance

Using the complex-step derivative approximation:

$$f'(x) \approx \frac{\text{Im}(f(x + ih))}{h}$$

with $h = 10^{-16}$, we obtained the sensitivity matrix $\chi$.

With $n = 51$ data points and $p = 3$ parameters:

$$\sigma^2 = \frac{1}{n-p} \mathbf{R}^\top \mathbf{R} = 426.7780$$

**Covariance Matrix:**

$$V = \sigma^2 (\chi^\top \chi)^{-1} = \begin{bmatrix} 1.7357 \times 10^{-8} & 1.8575 \times 10^{-7} & 1.4540 \times 10^{-7} \\ 1.8575 \times 10^{-7} & 2.6153 \times 10^{-5} & 3.1790 \times 10^{-5} \\ 1.4540 \times 10^{-7} & 3.1790 \times 10^{-5} & 8.4797 \times 10^{-5} \end{bmatrix}$$

**Correlation Coefficients:**

$$\rho_{\gamma,\delta} = 0.2757, \quad \rho_{\gamma,r} = 0.1198, \quad \rho_{\delta,r} = 0.6751$$

The parameters $\delta$ and $r$ are correlated, but others are weakly correlated. The covariance matrix has rank 3, which is expected from Project 1.

### Part (c): Parameter Distributions

**Standard Deviations:**

$$\sigma_\gamma = 1.3174 \times 10^{-4}, \quad \sigma_\delta = 0.0051, \quad \sigma_r = 0.0092$$

**95% Confidence Intervals:**

$$\gamma \in [0.0097, 0.0103]$$

$$\delta \in [0.1850, 0.2056]$$

$$r \in [0.7785, 0.8155]$$

### Part (d): Influenza Outbreak Analysis

For the British boarding school influenza data with $N = 763$, $S(0) = 760$, $I(0) = 3$, $R(0) = 0$, and $\delta = 0$:

| Day | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Confined | 3 | 8 | 26 | 76 | 225 | 298 | 258 | 233 | 189 | 128 | 68 | 29 | 14 | 4 |

**Table 2:** Influenza data

**Estimated Parameters:**

$$\gamma = 0.0022, \quad r = 0.4469$$

With $n = 14$ points and $p = 2$ parameters:

$$\sigma^2 = 325.4893$$

**Covariance Matrix:**

$$V = \begin{bmatrix} 3.3117 \times 10^{-10} & 1.3030 \times 10^{-7} \\ 1.3030 \times 10^{-7} & 1.5355 \times 10^{-4} \end{bmatrix}$$

**Correlation:** $\rho_{\gamma,r} = 0.5778$

**Standard Deviations:**

$$\sigma = 18.0413, \quad \sigma_\gamma = 1.8198 \times 10^{-5}, \quad \sigma_r = 0.0124$$

**95% Confidence Intervals:**

$$\gamma \in [0.00217, 0.00226], \quad r \in [0.4215, 0.4755]$$

---

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Heat model parameter estimation |
| Problem2.m / Problem2.py | Helmholtz energy OLS estimation |
| Problem3.m / Problem3.py | SIR model parameter estimation |

## References

1. Smith, R.C. (2013). *Uncertainty Quantification: Theory, Implementation, and Applications*. SIAM.
2. Banks, H.T. and Tran, H.T. (2009). *Mathematical and Experimental Modeling of Physical and Biological Processes*. CRC Press.
