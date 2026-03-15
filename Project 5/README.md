# Project 5: Surrogate Models

**Math 540: Uncertainty Quantification**

**Author:** Tilekbek Zhoroev

---

## Problem 1: Polynomial Surrogate with Latin Hypercube Sampling

### Problem Statement

Consider the function:

$$f(q) = (6q^2 + 3)\sin(6q - 4)$$

for $q \in [0, 2]$. Plot the function and $M = 15$ random samples $q^m \sim U(0, 2)$ and Latin hypercube samples. For the Latin hypercube samples in qdata.txt, use regression to construct an 8th-order polynomial surrogate $f_s^K(q)$ and compare to $f(q)$ for $q \in [0, 2]$ and $q \in [-0.5, 2.5]$.

### Solution

#### Random vs Latin Hypercube Sampling

We use `rand.m` to generate random samples from $U(0, 1)$, rescaled to $(0, 2)$ using:

$$a + (b - a)\eta, \quad \eta \in (0, 1)$$

with $a = 0$, $b = 2$. Similarly, using `lhsdesign.m` and the same transformation, we obtain Latin hypercube samples.

**Key Observations:**
- Random samples tend to cluster, reducing surrogate accuracy
- Latin hypercube samples fill the entire interval without clustering while maintaining randomness

#### Polynomial Surrogate Construction

Using Latin hypercube samples, we compute:

$$y = [y^1, \ldots, y^M]^\top, \quad y^i = f(q^i), \quad i = 1, \ldots, M$$

The polynomial surrogate coefficients $u_k$ in:

$$f_s^K(q) = \sum_{k=0}^{K} u_k q^k$$

are computed by minimizing the least squares functional:

$$J(u) = \sum_{m=0}^{M} \left[y^m - \sum_{k=0}^{K} u_k (q^m)^k\right]^2 = (y - Xu)^\top (y - Xu)$$

where the design matrix is:

$$X = \begin{bmatrix} 1 & q^1 & (q^1)^2 & \cdots & (q^1)^K \\ 1 & q^2 & (q^2)^2 & \cdots & (q^2)^K \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & q^M & (q^M)^2 & \cdots & (q^M)^K \end{bmatrix}$$

with $M = 15$ samples and $K = 8$ polynomial order.

The optimal solution is:

$$u = (X^\top X)^{-1} X^\top y$$

#### Results

The polynomial surrogate is accurate in the calibration region $[0, 2]$, but performs extremely poorly for extrapolatory predictions in $[-0.5, 2.5]$. This demonstrates the danger of using polynomial surrogates for out-of-data predictions.

![Sampling Comparison](Figures&Data/11.eps)

**Figure 1:** (a) Function, random samples, and Latin hypercube samples; Polynomial surrogate for (b) $q \in [0, 2]$; (c) $q \in [-0.5, 2.5]$.

---

## Problem 2: Legendre Polynomial Surrogate

### Problem Statement

Consider the spring model:

$$\frac{d^2 z}{dt^2} + kz = 0, \quad z(0) = 3, \quad \frac{dz}{dt}(0) = 0$$

with solution $z(t, k) = 3\cos(\sqrt{k} \cdot t)$.

We consider $k \sim U(\bar{k} - \sigma_k, \bar{k} + \sigma_k)$ with $\bar{k} = 8.5$ and $\sigma_k = 0.001$. Construct a Legendre surrogate $f_s^K(t, \xi)$ using $K + 1 = 4$ basis functions $\{\psi_k(\xi)\}_{k=0}^K$ with $\xi \in [-1, 1]$.

### Solution

Define the mapping $g(\xi): [-1, 1] \to [\bar{k} - \sigma_k, \bar{k} + \sigma_k]$:

$$g(\xi) = \frac{a + b}{2} + \frac{b - a}{2}\xi$$

where $a = \bar{k} - \sigma_k$, $b = \bar{k} + \sigma_k$.

The Legendre polynomial surrogate is:

$$f_s^K(t, k) = f_s^K(t, g(\xi)) = \sum_{k=0}^{K} u_k(t)\psi_k(\xi)$$

#### Coefficient Computation

Using discrete projections:

$$u_k(t) = \frac{1}{\gamma_k} \int_{-1}^{1} f(t, g(\xi))\psi_k(\xi)\rho(\xi) \, d\xi$$

$$= \frac{1}{\gamma_k} \int_{-1}^{1} 3\cos(\sqrt{g(\xi)} \cdot t)\psi_k(\xi)\rho(\xi) \, d\xi$$

$$\approx \frac{1}{\gamma_k} \sum_{r=1}^{R} 3\cos(\sqrt{g(\xi_r)} \cdot t)\psi_k(\xi_r)\omega_r$$

for $k = 0, \ldots, K$, where $\xi_r$ and $\omega_r$ are Legendre quadrature points and weights ($K = 3$, $R = 10$).

#### Mean and Standard Deviation

**Surrogate formulas:**

$$E[f_s^K(t, \xi)] = u_0(t)$$

$$\sqrt{\text{var}[f_s^K(t, \xi)]} = \left[\sum_{k=1}^{K} u_k^2(t)\gamma_k\right]^{1/2}$$

**Monte Carlo approximations** with $M = 10^5$ samples:

$$E[f(t, k)] = \frac{1}{M} \sum_{m=1}^{M} f(t, k^m)$$

$$\sqrt{\text{var}[f_s^K(t, \xi)]} = \left[\frac{1}{M-1} \sum_{m=1}^{M} [f(t, k^m) - E[f(t, k)]]^2\right]^{1/2}$$

#### Results

Both computational methods (discrete projection and Monte Carlo) produce identical results for mean and standard deviation.

![Legendre Surrogate Results](Figures&Data/12.eps)

**Figure 2:** (a) Mean and (b) standard deviation computed using Legendre surrogate with discrete projection and Monte Carlo sampling.

---

## Problem 3: Gaussian Process Surrogate

### Problem Statement

For the function $f(q) = (6q^2 + 3)\sin(6q - 4)$ and $M = 15$ Latin hypercube samples from qdata.txt, use `fitrgp.m` to construct a Gaussian process (GP) surrogate using a squared exponential kernel.

### Solution

#### Squared Exponential Kernel

We use the squared exponential covariance kernel:

$$c(q, q') = \sigma^2 e^{-(q-q')^2 / 2l^2}$$

with constant mean function $\mu(q) = \mu_0$.

#### Hyperparameter Optimization

Using $M = 15$ training points and `fitrgp.m`, we obtain optimal hyperparameters:

$$\sigma = 16.9326, \quad \mu_0 = 2.6996, \quad l = 0.3491$$

#### Results

**Calibration region $[0, 2]$:** The GP provides accurate predictions with tight confidence intervals near training points.

**Extrapolation region $[-0.5, 2.5]$:** Out-of-data predictions are better than polynomial surrogates. The GP prediction reverts to the prior mean with increased uncertainty, which is more physically reasonable than polynomial extrapolation.

![GP Calibration](Figures&Data/4.eps)

![GP Extrapolation](Figures&Data/5.eps)

**Figure 3:** (a), (c) Covariance function; (b), (d) Training data, mean, and 95% predictive distribution for $q \in [0, 2]$ and $q \in [-0.5, 2.5]$.

---

## Summary

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| Polynomial | Simple, fast | Poor extrapolation, requires many terms |
| Legendre | Orthogonal basis, stable | Limited flexibility |
| Gaussian Process | Uncertainty quantification, better extrapolation | Computationally expensive for large datasets |

---

## Code Files

| File | Description |
|------|-------------|
| Problem1.m / Problem1.py | Polynomial surrogate with LHS |
| Problem2.m / Problem2.py | Legendre polynomial surrogate |
| Problem3.m / Problem3.py | Gaussian process regression |

## References

1. Rasmussen, C.E. and Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
2. Xiu, D. (2010). *Numerical Methods for Stochastic Computations*. Princeton University Press.
3. Smith, R.C. (2013). *Uncertainty Quantification*. SIAM.
