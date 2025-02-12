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
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=64'

import sys

import jax

import jax.numpy as jnp
import jax.random as jr
import time

from kozax.genetic_programming import GeneticProgramming
from pysr import PySRRegressor

def clock_pysr(seed, N):
    x = jnp.linspace(0, 20, N)[:,None]
    y = x**2 - x/3 + 1

    start = time.time()

    model = PySRRegressor(
            binary_operators="+ * - / ^".split(" "),
            unary_operators=[],
            loss="loss(x, y) = abs(x-y)",
            constraints={"^": (-1, 3)},
            equation_file=f"results/output.csv",
            population_size=64,
            niterations=50,
            populations=15,
            multithreading=True,
            verbosity=0
            )

    model.fit(x, y)
    end = time.time()

    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    return elapsed_time

def clock_kozax(seed, N, backend):
    x = jnp.linspace(0, 20, N)[:,None]
    y = x**2 - x/3 + 1

    class FitnessFunction:
        def __call__(self, candidate, data, tree_evaluator):
            _X, _Y = data
            pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, _X)
            errors = jnp.abs(pred-_Y)/jnp.mean(jnp.abs(_Y))
            nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(errors)
            errors = jnp.where(nan_or_inf, jnp.ones(errors.shape)*1e8, errors)
            fitness = jnp.mean(errors)
            return jnp.clip(fitness,0,1e8)

    population_size = 64
    num_populations = 15
    num_generations = 50

    fitness_function = FitnessFunction()

    layer_sizes = jnp.array([1])

    operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                    ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
                    ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                    ("**", lambda x, y: jnp.power(x, y), 2, 0.1), 
                    ("/", lambda x, y: jnp.divide(x, y), 2, 0.1)
                    ]

    variable_list = [["x"]]

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations,
                            max_nodes = 15, coefficient_optimisation="ES", start_coefficient_optimisation = 0, optimise_coefficients_elite=64, ES_n_iterations=2, 
                            ES_n_offspring = 15, migration_percentage=0.125, backend = backend)

    key = jr.PRNGKey(seed)
    key, init_key = jr.split(key)

    start = time.time()

    population = strategy.initialize_population(init_key)

    for g in range(num_generations):
        key, eval_key, sample_key = jr.split(key, 3)
        fitness, population = strategy.evaluate_population(population, (x, y), eval_key)

        if g < (num_generations-1):
            population = strategy.evolve(population, fitness, sample_key)

    end = time.time()
    strategy.print_pareto_front()

    elapsed_time = end - start
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    return elapsed_time

if __name__ == '__main__':
    seed = 0
    print(seed)
    method = "Kozax"
    # method = "PySR"

    backend = "cpu"
    # backend = "gpu"

    N = 1000

    if method == "PySR":
        time = clock_pysr(seed, N)
    elif method == "Kozax":
        time = clock_kozax(seed, N, backend)

    import numpy as np
    np.save(f'data/runtime/{method}_{backend}/{N}/{seed}', time)  