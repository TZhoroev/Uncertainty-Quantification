# Uncertainty-Quantification

My solutions for MA 540: Uncertainty Quantification offered by NCSU.

This topics covers the following topics:
  * Determine the sources and impacts of input and response uncertainties in models arising in their discipline as well as prototypical weather, climate, hydrology, nuclear and biology models.
  * Explain the basic probability, stochastic process and statistics concepts required for uncertainty quantification.
  * Formulate models in a manner that isolates the influential parameters and facilitates statistical analysis. This includes the use of local and global sensitivity analysis techniques.
  * Construct surrogate models for complex processes that retain the fundamental underlying behavior while providing the computational efficiency required for model calibration and uncertainty propagation.
  * Compute confidence intervals using frequentist analysis and employ Markov chain Monte Carlo methods to construct posterior distributions and credible intervals for parameters. Be able to verify the accuracy of distributions constructed using Bayesian analysis.
  * Compute confidence, credible and prediction intervals for model responses and quantities of interest using sampling techniques and numerical stochastic spectral methods.
  
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
 
  - Learned how to use the provided observational data to calculate the model's parameters and parameter distributions.
  - Create and solve a constrained or unconstrained optimization problem based on the physical and biological constraints of the parameters. 

 1. [Problem 1: Using non-linear constraint optimization find the parameters of the heat model for copper rod.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem1.m)
 2. [Problem 2: Compute OLS estimate for the model parameters and variance of the Helmholtz energy model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem2.m)
 3. [Problem 3: Find parameter distributions of the SIR model.](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Problem3.m)
 * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%202/Project_2_writeup.pdf)*
 
 
 [Project 3](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%202):
 
  - Learned how to obtain the parameter distributions using Bayesian methods.
  - Discovered the theory and application of Markov Chain Monte Carlo methods for calculating posterior distributions. The advantages and disadvantages of this method in comparison to other adaptive methods.
  - The following posterior estimation algorithms are covered:
     -  Metropolis, Hamiltonian Monte Carlo, the Metropolis-adjusted Langevin algorithm, Delayed Rejection Adaptive Metropolis, etc., and their use cases. 
  - The failure cases of the posterior sampling algorithms and Bayesian parameter estimation are covered.
  - How to Apply Bayesian Methods to find the identifiable subset of parameters.
  
  
 1. [Problem 1: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem1.m)
 2. [Problem 2: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem2.m)
 3. [Problem 3: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Problem3.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%203/Project_3_writeup.pdf)*
  
  
  [Project 4](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%204):
 
 1. [Problem 1: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem1.m)
 2. [Problem 2: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem2.m)
 3. [Problem 3: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem3a.m)
 4. [Problem 4: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Problem4.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%204/Project_4_writeup.pdf)*
  
  
   [Project 5](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%205):
 
 1. [Problem 1: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem1.m)
 2. [Problem 2: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem2.m)
 3. [Problem 3: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Problem3.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%205/Project_5_writeup.pdf)*
  
     [Project 6](https://github.com/TZhoroev/Uncertainty-Quantification/tree/main/Project%206):
 
 1. [Problem 1: ](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%206/Final.m)
  * *[Project Writeup](https://github.com/TZhoroev/Uncertainty-Quantification/blob/main/Project%206/Project_6_writeup.pdf)*
  
  
