# Kozax: Fast and flexible Genetic Programming in JAX

In this repository, you can find the source code and data used for the paper "Kozax: Fast and flexible Genetic Programming in JAX". Click PLACEHOLDER to read the paper. 

## Build
To use the code, you can clone the repository and create the environment by running:
<!-- ```
conda env create -f environment.yml
conda activate gp_policies
``` -->

'src' contains the scripts for reproducing the experiments. 'data' contains the Pareto fronts of Kozax and PySR for each experiment.

|Experiment|Code|PySR|Kozax|
|---|---|---|---|
|Kepler's third law| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/law_discovery.py)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/PySR_results/Kepler)|[Data]()|
|Newton's law of universal gravitation| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/law_discovery.py)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/PySR_results/Newton)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/Newton)|
|Bode's law| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/law_discovery.py)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/PySR_results/Bode)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/Bode)|
|Fully observable Lotka-Volterra equations| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/finite_differences_method.py)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/PySR_results/LV_full)|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/LV_full)|
|Partially observable Lotka-Volterra equations| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/ODE_integration.py)|-|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/LV_partial)|
|Acrobot| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/symbolic_policy.py)|-|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/Acrobot)|
|Loss function| [Code](https://github.com/sdevries0/kozax_paper/blob/main/src/loss_function_optimization.py)|-|[Data](https://github.com/sdevries0/kozax_paper/tree/main/data/Kozax_results/Loss_function)|


<!-- ## Citation
If you make use of this code in your research paper, please cite:
```
PLACEHOLDER
``` -->
