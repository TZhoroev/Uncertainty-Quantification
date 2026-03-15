# Project 4: Uncertainty Propagation

**Math 540: Uncertainty Quantification**

**Author:** Tilekbek Zhoroev

---

## Problem 1: Height-Weight Model Prediction Intervals

### Problem Statement

Consider the height-weight model:

$$Y_i = \theta_0 + \theta_1 (x_i/12) + \theta_2 (x_i/12)^2 + \varepsilon_i$$

with the data in Table 1. Using the parameter and covariance matrix estimates, compute and plot $2\sigma$ and frequentist prediction intervals for heights ranging from 58 to 72 inches. Repeat for the extrapolatory regime of 50 to 80 inches.

| Height (in) | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
|-------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|---| 
| Weight (lbs) | 115 | 117 | 120 | 123 | 126 | 129 | 132 | 135 | 139 | 142 | 146 | 150 | 154 | 159 | 164 |

**Table 1:** Height-weight data

### Given Parameters

$$\theta = \begin{bmatrix} 261.88 \\ -88.18 \\ 11.96 \end{bmatrix}, \quad V = \begin{bmatrix} 634.88 & -235.04 & 21.66 \\ -235.04 & 87.09 & -8.03 \\ 21.66 & -8.03 & 0.74 \end{bmatrix}$$

### Solution

The design matrix is:

$$X = \begin{bmatrix} 1 & x_1/12 & (x_1/12)^2 \\ 1 & x_2/12 & (x_2/12)^2 \\ \vdots & \vdots & \vdots \\ 1 & x_n/12 & (x_n/12)^2 \end{bmatrix}$$

### Response Intervals

Since $E[Y] = X\theta_0$ and $\text{cov}[Y] = XVX^\top + V_{obs}$, where $V_{obs} = \sigma^2 I_n$, the $\pm 2\sigma_Y$ interval is:

$$X_i\theta_0 \pm 2\sqrt{(\text{cov}[Y])_{ii}}$$

where $X_i$ is the $i$-th row of the design matrix.

### Frequentist Prediction Interval

The $(1-\alpha) \times 100\%$ frequentist prediction interval at $x_*$ is:

$$\hat{Y}_{x_*} \pm t_{n-p, 1-\alpha/2} \cdot \hat{\sigma} \sqrt{1 + x_* (X^\top X)^{-1} x_*^\top}$$

### Results

**Calibration Domain [58, 72]:** Both the $\pm 2\sigma_Y$ and frequentist 95% prediction intervals are nearly identical, and all data points fall within the intervals.

**Extrapolation Domain [50, 80]:** The prediction intervals are significantly wider than in the calibration domain, demonstrating increased uncertainty when extrapolating beyond the observed data range.

![Calibration Domain](Figures/1_1.eps)

![Extrapolation Domain](Figures/1_2.eps)

**Figure 1:** Data, $2\sigma_Y$ and prediction intervals for (a) calibration domain [58, 72] and (b) extrapolation domain [50, 80].

---

## Problem 2: Heat Model Bayesian Intervals (Aluminum Rod)

### Problem Statement

Consider the steady state heat model:

$$\frac{d^2 u_s(x)}{dx^2} = \frac{2(a+b)h}{kab}[u_s(x) - u_{amb}], \quad 0 < x < L$$

with boundary conditions:

$$\frac{du_s}{dx}(0) = \frac{\Phi}{k}, \quad \frac{du_s}{dx}(L) = \frac{h}{k}[u_{amb} - u_s(L)]$$

Use a subset of data (Table 3) to construct prediction intervals that extrapolate beyond the calibration domain. Use $k = 2.37$ for aluminum.

| x (cm) | 10 | 14 | 18 | 22 | 26 | 30 | 34 | 38 | 42 | 46 | 50 | 54 | 58 | 62 | 66 |
|--------|----|----|----|----|----|----|----|----|----|----|----|----|----|----|---|
| Temp | 96.14 | 80.12 | 67.66 | 57.96 | 50.90 | 44.84 | 39.75 | 36.16 | 33.31 | 31.15 | 29.28 | 27.88 | 27.18 | 26.40 | 25.86 |

**Table 2:** Complete steady-state temperature data for aluminum rod.

| x (cm) | 22 | 26 | 30 | 34 | 38 | 42 | 46 | 50 | 54 |
|--------|----|----|----|----|----|----|----|----|---|
| Temp | 57.96 | 50.90 | 44.84 | 39.75 | 36.16 | 33.31 | 31.15 | 29.28 | 27.88 |

**Table 3:** Calibration data subset.

### Solution

Using DRAM with parameter estimates from Example 12.17, all chains converge. The chain covariance matrix:

$$V = \begin{bmatrix} 1.3538 \times 10^{-1} & -1.0206 \times 10^{-5} \\ -1.0206 \times 10^{-5} & 7.9377 \times 10^{-10} \end{bmatrix}$$

Variance: $\sigma^2 = 0.0461$

Using `mcmcpred` and `mcmcpredplot`, we construct 95% credible and prediction intervals. Due to the small error variance and parameter variance, both intervals are narrow.

![Parameter Densities](Figures/4.eps)

**Figure 2:** Densities for (a) $\Phi$; (b) $h$; (c) $\varepsilon$.

![Heat Prediction Intervals](Figures/5.eps)

**Figure 3:** Data, 95% credible and prediction intervals for aluminum rod.

---

## Problem 3: SIR Model Credible Intervals

### Problem Statement

Consider the SIR model:

$$\frac{dS}{dt} = \delta N - \delta S - \gamma IS, \quad S(0) = 900$$

$$\frac{dI}{dt} = \gamma IS - (r + \delta)I, \quad I(0) = 100$$

$$\frac{dR}{dt} = rI - \delta R, \quad R(0) = 0$$

where $\gamma, r, \delta \in [0, 1]$.

### Part (a): SIR.txt Data

Using DRAM from Project 3 (with converged chains), we construct 95% credible and prediction intervals for $I(t)$ using `mcmcpred` and `mcmcpredplot`.

### Part (b): Influenza Data

For the British boarding school data with $\delta = 0$, $S(0) = 730$, $I(0) = 3$, $R(0) = 0$, $N = 733$:

| Day | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Confined | 3 | 8 | 26 | 76 | 225 | 298 | 258 | 233 | 189 | 128 | 68 | 29 | 14 | 4 |

**Table 4:** Influenza data

Using converged DRAM chains, we obtain 95% credible and prediction intervals.

![SIR Prediction Intervals](Figures/3_d1.jpg)

![Influenza Prediction Intervals](Figures/3_d2.jpg)

**Figure 4:** Data, 95% credible and prediction intervals: (a) SIR.txt data; (b) Influenza data.

---

## Problem 4: Frequentist vs Bayesian Comparison

### Problem Statement

For the SIR model with parameters $\theta = [\gamma, r, \delta]$ and nominal values $\bar{\theta} = [0.0100, 0.7970, 0.1953]$, estimate the sensitivity matrix $S$ with entries:

$$[S]_{ij} = \frac{\partial I}{\partial \theta_j}(t_i, \bar{\theta})$$

Use $\sigma^2 = 426.8$ to construct response and observation variance matrices.

### Solution

Using the complex-step approximation:

$$f'(x) \approx \frac{\text{Im}(f(x + ih))}{h}$$

we compute sensitivities and the sensitivity matrix $S$. Then:

**Response variance:**
$$\text{var}[f(\theta)] = SVS^\top$$

**Observation variance:**
$$\text{var}[Y] = SVS^\top + \sigma^2 I_{n \times n}$$

Both are $51 \times 51$ matrices (n = 51 data points).

### Interval Construction

The mean response is $I(t_i, \bar{\theta})$.

**$\pm 2\sigma_f$ interval:**
$$I(t_i, \bar{\theta}) \pm 2\sqrt{(\text{var}[f(\theta)])_{ii}}$$

**$\pm 2\sigma_Y$ interval:**
$$I(t_i, \bar{\theta}) \pm 2\sqrt{(\text{var}[Y])_{ii}}$$

### Comparison

The frequentist intervals are slightly narrower than the Bayesian 95% credible and prediction intervals, but the difference is small. This indicates that the linearization approximation is accurate for this problem.

![Frequentist Intervals](Figures/6.eps)

**Figure 5:** Data, $\pm 2\sigma_f$ and $\pm 2\sigma_Y$ intervals using SIR.txt data.

---

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Height-weight prediction intervals |
| Problem2.m / Problem2.py | Heat model Bayesian intervals |
| Problem3a.m / Problem3.py | SIR credible intervals |
| Problem4.m / Problem4.py | Frequentist vs Bayesian comparison |

## References

1. Wasserman, L. (2004). *All of Statistics*. Springer.
2. Gelman, A. et al. (2013). *Bayesian Data Analysis*. CRC Press.
3. Smith, R.C. (2013). *Uncertainty Quantification*. SIAM.
