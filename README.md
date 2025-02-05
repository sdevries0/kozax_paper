# Kozax: Flexible and Scalable Genetic Programming in JAX

In this repository, you can find the source code and data used for the paper "Kozax: Flexible and Scalable Genetic Programming in JAX". Click [PLACEHOLDER FOR ANONYMOSITY] to read the paper. 

## Build
To use the code, you can clone the repository and create the environment by running:
```
conda env create -f environment.yml
conda activate kozax_paper
```

`src` contains the scripts for reproducing the experiments. `data` contains the Pareto fronts of Kozax and PySR for each experiment.

|Experiment|Code|PySR|Kozax|
|---|---|---|---|
|Kepler's third law| [Code](src/law_discovery.py)|[Data](data/PySR_results/Kepler)|[Data]()|
|Newton's law of universal gravitation| [Code](src/law_discovery.py)|[Data](data/PySR_results/Newton)|[Data](data/Kozax_results/Newton)|
|Bode's law| [Code](src/law_discovery.py)|[Data](data/PySR_results/Bode)|[Data](data/Kozax_results/Bode)|
|Fully observable Lotka-Volterra equations| [Code](src/finite_differences_method.py)|[Data](data/PySR_results/LV_full)|[Data](data/Kozax_results/LV_full)|
|Partially observable Lotka-Volterra equations| [Code](src/ODE_integration.py)|-|[Data](data/Kozax_results/LV_partial)|
|Acrobot| [Code](src/symbolic_policy.py)|-|[Data](data/Kozax_results/Acrobot)|
|Loss function| [Code](src/loss_function_optimization.py)|-|[Data](data/Kozax_results/Loss_function)|

## Citation
If you make use of this code in your research paper, please cite:
```
[PLACEHOLDER FOR ANONYMOSITY]
```
