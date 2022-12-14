# Uncertainty-Quantification

![Plot](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/UQ.png)

This course covers the following topics :
  * Determine the sources and impacts of input and response uncertainties in models arising in their discipline as well as prototypical weather, climate, hydrology, nuclear and biology models.
  * Explain the basic probability, stochastic process and statistics concepts required for uncertainty quantification.
  * Formulate models in a manner that isolates the influential parameters and facilitates statistical analysis. This includes the use of local and global sensitivity analysis techniques.
  * Construct surrogate models for complex processes that retain the fundamental underlying behavior while providing the computational efficiency required for model calibration and uncertainty propagation.
  * Compute confidence intervals using frequentist analysis and employ Markov chain Monte Carlo methods to construct posterior distributions and credible intervals for parameters. Be able to verify the accuracy of distributions constructed using Bayesian analysis.
  * Compute confidence, credible and prediction intervals for model responses and quantities of interest using sampling techniques and numerical stochastic spectral methods.
  
  - ![#f03c15](https://placehold.co/15x15/f03c15/f03c15.png) Required Toolboxes: [Optimization Toolbox](https://www.mathworks.com/products/optimization.html) and [Global Optimization Toolbox](https://www.mathworks.com/products/global-optimization.html)
  - ![#c5f015](https://placehold.co/15x15/c5f015/c5f015.png) Required Package: [MCMC for MATLAB](https://mjlaine.github.io/mcmcstat/)
  
 [Project 1](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%201): 
 
  - Learned the various methods, including analytical, finite difference, and complex-step approximations, to compute the sensitivity equations of the models with respect to each parameter of the model. 
  - Creating and employing the Fisher (information) matrix to perform local sensitivity analysis and identify an identifiable subset of parameters. 
  - Studied different statistical distances to compare the distributions. 
  - Covered different algorithm to perform global sensitivity analysis to the model.
 
 1. [Problem 1: Compute sensitivities of spring model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%201/UQ_8_5.m)
 2. [Problem 2: Compute sensitivities of SIR model and find identifible subset of parameters.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%201/UQ_8_8.m)
 3. [Problem 3: Find identifible subset of parameters of the heat equation using given observation data.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%201/UQ_8_9.m)
 4. [Problem 4: Apply global sensitivity algorithms: Morris screening, Sobol indexes and Saltelli algorithm for Helmholtz energy model](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%201/UQ_9_6.m)
 * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%201/Project_1_writeup.pdf)*
 
 [Project 2](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%202)
 
  - Obtain the identifiable subsets using gradient bases activa subspace contructions.
  - Learned how to use the provided observational data to calculate the model's parameters and parameter distributions.
  - Create and solve a constrained or unconstrained optimization problem based on the physical and biological constraints of the parameters.??

 1. [Problem 1: Using non-linear constraint optimization find the parameters of the heat model for copper rod.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem1.m)
 2. [Problem 2: Compute OLS estimate for the model parameters and variance of the Helmholtz energy model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem2.m)
 3. [Problem 3: Find parameter distributions of the SIR model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem3.m)
 * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Project_2_writeup.pdf)*
 
 
 [Project 3](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%202):
 
  - Learned how to obtain the parameter distributions using Bayesian methods.
?? - Disscussed the theory and application of Markov Chain Monte Carlo methods for calculating posterior distributions. The advantages and disadvantages of this method in comparison to other adaptive methods.
  - The following posterior estimation algorithms are covered:
     -  Metropolis, Hamiltonian Monte Carlo, the Metropolis-adjusted Langevin algorithm, Delayed Rejection Adaptive Metropolis, etc., and their use cases.??
  - The failure cases of the posterior sampling algorithms and Bayesian parameter estimation are covered.
  - How to Apply Bayesian Methods to find the identifiable subset of parameters.
  
  
 1. [Problem 1: Compare the posterior distributions of the parameters for the heat equation.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem1.m)
 2. [Problem 2: Find the parameters using optimization and Bayesian optimization. Compare the results and identify the issues with the algebraic dependency on MCMC algorithms.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem2.m)
 3. [Problem 3: Obtain the parameter distribution using different algorithms for the Helmholtz energy model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem3.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Project_3_writeup.pdf)*
  
  
  [Project 4](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%204):
  
   - In previous lectures, we covered how to determine the distributions for parameters. Hence, the next reasonable question is, "How do we efficiently propagate input uncertainties through models?"
   - Using uncertainty propagation, we obtained the mean response, credible intervals, and prediction intervals for the quantity of interest..
   - We learned sampling and perturbation methods for uncertainty propagation in frequentist and Bayesian techniques.
 
 1. [Problem 1: Obtain the frequentist confidence and prediction intervals for the height-weight model in the calibration and extrapolation domains.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem1.m)
 2. [Problem 2: Obtain the Bayesian credible and prediction intervals for the heat model of the aluminum rod in the calibration domain.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem2.m)
 3. [Problem 3: Obtain the Bayesian credible and prediction intervals for the SIR model in the calibration domain using different observational datasets.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem3a.m)
 4. [Problem 4: Obtain the frequentist confidence and prediction intervals for the SIR model in the calibration domain using different observational datasets. Then, compare the results with the previous problem's results.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem4.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Project_4_writeup.pdf)*
  
  
   [Project 5](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%205):
   
   
   - We covered the techniques for surrogate and reduced-order models.
   - General polynomial methods: Sparse grid techniques and importance of the latin hypercube sampling for surrogate models.
   - Stochastic Galerkin Methods: Projection methods, the algorithms are available in Sandia Dakota package for high dimensional models.
   - Statistical Surrogate Models: Gaussian process or Kriging representations, Neural Networks.
 
 1. [Problem 1: Find the 8th - order polynomial surrogate model using random sampling and Latin hypercube samplings. Discuss the limitations on extrapolation domains. ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem1.m)
 2. [Problem 2: Find the Legendre surrogate model using Discrete projection and Monte Carlo methods. ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem2.m)
 3. [Problem 3: Find the Gaussian surraget model of given dataset.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem3.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Project_5_writeup.pdf)*
  
 [Project 6](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%206):
 
  - Quantification of the physical and surrogate model discrepancies
 
 1. [Problem 1: Explore the Dittus-Boelter equation](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%206/Final.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%206/Project_6_writeup.pdf)*
  
  
