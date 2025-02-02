"""
Kozax: Flexible and Scalable Genetic Programming in JAX

Copyright (c) 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import jax

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from kozax.genetic_programming import GeneticProgramming

from pysr import PySRRegressor
   
def get_data(dataset, key):
    if dataset == "Kepler":
        X = jnp.array([0.389, 0.724, 1, 1.524, 5.20, 9.510])
        Y = jnp.array([87.77, 224.70, 365.25, 686.95, 4332.62, 10759.2])

        return X[:,None], Y[:,None]

    elif dataset == "Newton":
        keys = jr.split(key, 5)
        G = jnp.log(6.67408e-11)
        N = 30
        r = jr.uniform(keys[0], shape = (N,), minval=jnp.log(3.7e8), maxval=jnp.log(7.37593e12))
        m1 = jr.uniform(keys[1], shape = (N,), minval=jnp.log(1.3e22), maxval=jnp.log(2e30))
        m2 = jr.uniform(keys[2], shape = (N,), minval=jnp.log(1.3e22), maxval=jnp.log(2e30))

        F = jnp.exp(G + m1 + m2 - 2*r)
        F += jr.normal(keys[3], shape=(N,))*0.1*F
        F = jnp.log(F)
        
        X = jnp.stack([r, m1, m2], axis=-1)
        Y = F[:,None]
                
        return X, Y

    elif dataset == "Bode":
        n = jnp.arange(-1,7,dtype=float)
        a = jnp.array([0.39, 0.72, 1.0, 1.52, 2.77, 5.2, 9.58, 19.22])
        X = n[:,None]
        Y = a[:,None]

        return X, Y
                   
def test_pysr(seeds, dataset, args, file_name, variable_list = ["x0"]):
    population_size, num_populations, num_generations = args

    for seed in seeds:
        key = jr.PRNGKey(seed)
        print(f"Seed: {seed}")
        key, data_key, init_key = jr.split(key, 3)

        X, Y = get_data(dataset, data_key)

        model = PySRRegressor(
            binary_operators="+ * - / ^".split(" "),
            loss="loss(x, y) = abs(x-y)",
            constraints={"^": (-1, 3)},
            equation_file=f"{file_name}/{seed}.csv",
            population_size=population_size,
            niterations=num_generations,
            populations=num_populations,
            random_state=seed,
            deterministic=True,
            procs=0,
            multithreading=False,
            verbosity=0
            )

        model.fit(X, Y, variable_names=variable_list)

class FitnessFunction:
    def __call__(self, candidate, data, tree_evaluator):
        _X, _Y = data
        pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, _X)
        errors = jnp.abs(pred-_Y)/jnp.mean(jnp.abs(_Y))
        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(errors)
        errors = jnp.where(nan_or_inf, jnp.ones(errors.shape)*1e8, errors)
        fitness = jnp.mean(errors)
        return jnp.clip(fitness,0,1e8)
    
def test_kozax(seeds, dataset, variable_list, operator_list, args, file_name, stepsize):
    population_size, num_populations, num_generations = args

    fitness_function = FitnessFunction()

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations = num_populations,
                            max_nodes = 15, coefficient_optimisation="ES", migration_period=5, size_parsimony=0.003, start_coefficient_optimisation = 0, 
                            optimise_coefficients_elite=500, ES_n_iterations=5, ES_n_offspring = 10, init_learning_rate=stepsize)
    
    for seed in seeds:
        strategy.reset()
        key = jr.PRNGKey(seed)
        print(f"Seed: {seed}")
        key, data_key, init_key = jr.split(key, 3)

        X, Y = get_data(dataset, data_key)

        population = strategy.initialize_population(init_key)

        for g in range(num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            fitness, population = strategy.evaluate_population(population, (X, Y), eval_key)

            if g < (num_generations-1):
                population = strategy.evolve(population, fitness, sample_key)

        strategy.print_pareto_front(save=True, file_name=f'{file_name}/{seed}')
        
if __name__ == '__main__':
    DATASETS = ["Kepler", "Newton", "Bode"]
    dataset_id = 2
    dataset = DATASETS[dataset_id]

    variable_lists = [
        [["X"]],
        [["r", "m1", "m2"]],
        [["X"]],
        ]
    
    variable_list = variable_lists[dataset_id]

    stepsizes = [0.2, 0.5, 0.2]
    
    operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1), 
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5), 
            ("**", lambda x, y: jnp.power(x, y), 2, 0.1), 
            ("/", lambda x, y: jnp.divide(x, y), 2, 0.1)
            ]

    args = [100, 10, 100] #size, num, generations

    seeds = np.arange(10)

    method = "Kozax"
    # method = "PySR"

    file_name = f'data/{method}_results/{dataset}'

    if method=="PySR":
        test_pysr(seeds, dataset, args, file_name, variable_list[0])

    elif method=="Kozax":
        test_kozax(seeds, dataset, variable_list, operator_list, args, file_name, stepsizes[dataset_id])