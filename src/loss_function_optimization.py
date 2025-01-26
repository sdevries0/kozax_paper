"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

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
import optax 

from kozax.genetic_programming import GeneticProgramming

class FitnessFunction:
    def __init__(self, input_dim, hidden_dim, output_dim, epochs, learning_rate):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.optim = optax.adam(learning_rate)
        self.epochs = epochs

    def __call__(self, candidate, data, tree_evaluator):
        data_keys, test_keys, network_keys = data
        losses = jax.vmap(self.train, in_axes=[None, 0, 0, 0, None])(candidate, data_keys, test_keys, network_keys, tree_evaluator)
        return jnp.mean(losses)

    def get_data(self, key, n_samples = 50):
        x = jr.uniform(key, shape=(n_samples, 2))
        y = jnp.logical_xor(x[:,0]>0.5, x[:,1]>0.5)

        return x, y[:,None]

    def loss_function(self, params, x, y, candidate, tree_evaluator):
        pred = self.neural_network(params, x)
        return jnp.mean(jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, jnp.concatenate([pred, y], axis=-1)))
    
    def train(self, candidate, data_key, test_key, network_key, tree_evaluator):
        params = self.init_network_params(network_key)

        optim_state = self.optim.init(params)

        def step(i, carry):
            params, optim_state, key = carry

            key, _key = jr.split(key)

            x_train, y_train = self.get_data(_key, n_samples=100)

            grads = jax.grad(self.loss_function)(params, x_train, y_train, candidate, tree_evaluator)
                
            # Update parameters
            updates, optim_state = self.optim.update(grads, optim_state, params)
            params = optax.apply_updates(params, updates)

            return (params, optim_state, key)

        (params, _, _) = jax.lax.fori_loop(0, self.epochs, step, (params, optim_state, data_key))

        x_test, y_test = self.get_data(test_key, n_samples=500)

        pred = self.neural_network(params, x_test)
        return 1 - jnp.mean(y_test==(pred>0.5))

    # Define the neural network function (forward pass)
    def neural_network(self, params, x):
        w1, b1, w2, b2, w3, b3 = params
        hidden = jnp.tanh(jnp.dot(x, w1) + b1)
        hidden = jnp.tanh(jnp.dot(hidden, w2) + b2)
        output = jnp.dot(hidden, w3) + b3
        return jax.nn.sigmoid(output)

    # Define the neural network model (1 hidden layer)
    def init_network_params(self, key):
        key1, key2, key3 = jr.split(key, 3)
        w1 = jr.normal(key1, (self.input_dim, self.hidden_dim)) * jnp.sqrt(2.0 / self.input_dim)
        b1 = jnp.zeros(self.hidden_dim)
        w2 = jr.normal(key2, (self.hidden_dim, self.hidden_dim)) * jnp.sqrt(2.0 / self.hidden_dim)
        b2 = jnp.zeros(self.hidden_dim)
        w3 = jr.normal(key3, (self.hidden_dim, self.output_dim)) * jnp.sqrt(2.0 / self.hidden_dim)
        b3 = jnp.zeros(self.output_dim)
        return (w1, b1, w2, b2, w3, b3)

def generate_keys(key, batch_size=4):
    key1, key2, key3 = jr.split(key, 3)
    return jr.split(key1, batch_size), jr.split(key2, batch_size), jr.split(key3, batch_size)

population_size = 100
num_populations = 10
num_generations = 50

seeds = jnp.arange(10)

operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                 ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
                 ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                 ("/", lambda x, y: jnp.divide(x, y + 1e-7), 2, 0.1),
                 ("**", lambda x, y: jnp.power(x, y), 2, 0.1),
                 ("log", lambda x: jnp.log(x + 1e-7), 1, 0.1),
                 ("exp", lambda x: jnp.exp(x), 1, 0.1)
                 ]

variable_list = [["pred", "y"]]

input_dim = 2
hidden_dim = 16
output_dim = 1

fitness_function = FitnessFunction(input_dim, hidden_dim, output_dim, learning_rate=0.01, epochs=500)

strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations = num_populations,
            max_nodes = 15, coefficient_optimisation="ES", migration_period=5, device_type='gpu',
            size_parsimony=0.003, optimise_coefficients_elite=50, ES_n_iterations=1, ES_n_offspring = 20, init_learning_rate=0.2)

for seed in seeds:
    strategy.reset()
    key = jr.PRNGKey(seed)
    print(f"Seed: {seed}")

    data_key, init_key = jr.split(key)
    data_keys, test_keys, network_keys = generate_keys(key)

    population = strategy.initialize_population(init_key)

    for g in range(num_generations):
        key, eval_key, sample_key = jr.split(key, 3)
        fitness, population = strategy.evaluate_population(population, (data_keys, test_keys, network_keys), eval_key)
        best_fitness, best_solution = strategy.get_statistics(g)
        print(f"In generation {g+1}, best fitness = {best_fitness:.4f}, best solution = {strategy.to_string(best_solution)}")

        if g < (num_generations-1):
            
            population = strategy.evolve(population, fitness, sample_key)

    strategy.print_pareto_front(save=True, file_name=f'data/Kozax_results/loss_f/{seed}')