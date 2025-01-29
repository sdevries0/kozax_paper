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
import gymnax

from kozax.genetic_programming import GeneticProgramming

class GymFitnessFunction:
    def __init__(self, env_name) -> None:
        self.env, self.env_params = gymnax.make(env_name)
        self.num_steps = 200

    def __call__(self, candidate, keys, tree_evaluator):
        reward = jax.vmap(self.simulate_trajectory, in_axes=(None, 0, None))(candidate, keys, tree_evaluator)
        return jnp.mean(reward)
        
    def simulate_trajectory(self, candidate, key, tree_evaluator):
        key, subkey = jr.split(key)
        state, env_state = self.env.reset(subkey, self.env_params)

        def policy(state):
            a = tree_evaluator(candidate, state)[0]
            return jax.lax.select(a == 0, 1, jax.lax.select(a > 0, 2, 0))

        def step_fn(carry, _):
            state, env_state, key = carry

            action = policy(state)

            key, subkey = jr.split(key)
            next_state, next_env_state, reward, _, _ = self.env.step(
                subkey, env_state, action, self.env_params
            )

            return (next_state, next_env_state, key), (state, reward)

        (_, (states, rewards)) = jax.lax.scan(
            step_fn, (state, env_state, key), None, length=self.num_steps
        )
        
        first_success = jnp.argmax(rewards)
        return (first_success + (first_success == 0) * self.num_steps)/self.num_steps

def evolve_control_policy():
    population_size = 100
    num_populations = 10
    num_generations = 50
    batch_size = 4

    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1), 
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5), 
        ("**", lambda x, y: jnp.power(x, y), 2, 0.1), 
        ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
        ("sin", lambda x: jnp.sin(x), 1, 0.1)
        ]

    fitness_function = GymFitnessFunction("Acrobot-v1")
    variable_list = [["y1", "y2", "y3", "y4", "y5", "y6"]]

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, num_populations = num_populations,
                            max_nodes = 15, coefficient_optimisation="ES", migration_period=5, size_parsimony=0.003, start_coefficient_optimisation = 0, 
                            optimise_coefficients_elite=50, ES_n_iterations=3, ES_n_offspring = 20, init_learning_rate=0.2)

    seeds = jnp.arange(10)

    for seed in seeds:
        strategy.reset()
        key = jr.PRNGKey(seed)
        print(f"Seed: {seed}")
        key, init_key, data_key = jr.split(key, 3)

        batch_keys = jr.split(data_key, batch_size)

        population = strategy.initialize_population(init_key)

        for g in range(num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            fitness, population = strategy.evaluate_population(population, (batch_keys), eval_key)

            if g < (num_generations-1):
                population = strategy.evolve(population, fitness, sample_key)

        strategy.print_pareto_front(save=True, file_name=f'data/Kozax_results/Acrobot/{seed}')

if __name__ == '__main__':
    evolve_control_policy()