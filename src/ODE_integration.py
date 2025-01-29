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
import diffrax
import jax.numpy as jnp
import jax.random as jr
import diffrax

from jax import Array
from typing import Tuple, Callable
from jax.random import PRNGKey

from kozax.genetic_programming import GeneticProgramming

class LotkaVolterra():
    def __init__(self):
        self.n_var = 2

        self.init_mu = jnp.array([10, 10])
        self.init_sd = 2

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4

    def sample_init_states(self, batch_size, key):
        return jr.uniform(key, shape = (batch_size,self.n_var), minval=5, maxval=15)
    
    def drift(self, t, state, args):
        return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])

class Evaluator:
    """Evaluator for candidates on symbolic regression tasks

    Attributes:
        dt0: Initial step size for integration
        fitness_function: Function that computes the fitness of a candidate
        system: ODE term of the drift function
        solver: Solver used for integration
        stepsize_controller: Controller for the stepsize during integration
        max_steps: The maximum number of steps that can be used in integration
    """
    def __init__(self, solver: diffrax.AbstractSolver = diffrax.Euler(), dt0: float = 0.01, max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize(), optimize_dimensions: Array = None) -> None:
        self.dt0 = dt0
        if optimize_dimensions is None:
            self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.abs(pred_ys-true_ys), axis=-1))/jnp.mean(true_ys) #Mean Absolute Error
        else:
            if len(optimize_dimensions) > 1:
                self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.sum(jnp.abs(pred_ys[:,optimize_dimensions]-true_ys[:,optimize_dimensions]), axis=-1))/jnp.mean(true_ys[:,optimize_dimensions])
            else:
                self.fitness_function = lambda pred_ys, true_ys: jnp.mean(jnp.abs(pred_ys[:,optimize_dimensions]-true_ys[:,optimize_dimensions]))/jnp.mean(true_ys[:,optimize_dimensions])

        self.system = diffrax.ODETerm(self._drift)
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.max_steps = max_steps

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the candidate
        """
        fitness, _ = self.evaluate_candidate(candidate, data, tree_evaluator)

        return jnp.mean(fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> Tuple[Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_time_series, in_axes=[None, 0, None, 0, None])(candidate, *data, tree_evaluator)
    
    def evaluate_time_series(self, candidate: Array, x0: Array, ts: Array, ys: Array, tree_evaluator: Callable) -> Tuple[Array, float]:
        """Solves the candidate as a differential equation and returns the predictions and fitness

        :param candidate: Candidate that is evaluated
        :param x0: Initial conditions of the environment
        :param ts: Timepoints of which the system has to be solved
        :param ys: Ground truth data used to compute the fitness
        :param process_noise_key: Key to generate process noise
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        
        saveat = diffrax.SaveAt(ts=ts)
        event_nan = diffrax.Event(self.cond_fn_nan)

        sol = diffrax.diffeqsolve(
            self.system, self.solver, ts[0], ts[-1], self.dt0, x0, args=(candidate, tree_evaluator), saveat=saveat, max_steps=self.max_steps, stepsize_controller=self.stepsize_controller, 
            adjoint=diffrax.DirectAdjoint(), throw=False, event=event_nan
        )
        pred_ys = sol.ys
        fitness = self.fitness_function(pred_ys, ys) + 1.0*jnp.mean(jnp.where(pred_ys<0, jnp.abs(pred_ys), 0))

        return fitness, pred_ys
    
    def _drift(self, t, x, args):
        candidate, tree_evaluator = args

        dx = tree_evaluator(candidate, x)
        return dx
    
    def cond_fn_nan(self, t, y, args, **kwargs):
        return jnp.where(jnp.any(jnp.isinf(y) + jnp.isnan(y)), True, False)


def get_data(key, env, dt, T, batch_size=20):
    x0s = env.sample_init_states(batch_size, key)
    ts = jnp.arange(0, T, dt)

    def solve(env, ts, x0):
        solver = diffrax.Dopri5()
        dt0 = 0.001
        saveat = diffrax.SaveAt(ts=ts)

        system = diffrax.ODETerm(env.drift)

        sol = diffrax.diffeqsolve(system, solver, ts[0], ts[-1], dt0, x0, saveat=saveat, max_steps=2000, 
                                  adjoint=diffrax.DirectAdjoint(), stepsize_controller=diffrax.PIDController(atol=1e-7, rtol=1e-7, dtmin=0.001))
        
        return sol.ys

    ys = jax.vmap(solve, in_axes=[None, None, 0])(env, ts, x0s)
    
    return x0s, ts, ys

def evolve_LV():
    T = 30
    dt = 0.2
    env = LotkaVolterra()

    operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.1), 
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5), 
            ("**", lambda x, y: jnp.power(x, y), 2, 0.1), 
            ("/", lambda x, y: jnp.divide(x, y), 2, 0.1)
        ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]

    population_size = 200
    num_populations = 10
    num_generations = 150

    fitness_function = Evaluator(solver=diffrax.Dopri5(), dt0 = 0.01, stepsize_controller=diffrax.PIDController(atol=1e-6, rtol=1e-6, dtmin=0.001), max_steps=300, optimize_dimensions = jnp.array([0]))

    layer_sizes = jnp.array([env.n_var])

    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations, backend='gpu',
                            max_nodes = 15, migration_period=5, coefficient_optimisation="ES", ES_n_offspring = 20, ES_n_iterations = 1, size_parsimony=0.003, 
                            start_coefficient_optimisation = 0, optimise_coefficients_elite=1000, init_learning_rate=0.1)

    seeds = jnp.arange(10)

    for seed in seeds:
        strategy.reset()

        key = jr.PRNGKey(seed)
        key, init_key, data_key = jr.split(key, 3)
        x0s, ts, ys = get_data(data_key, env, dt=dt, T=T, batch_size=8)

        population = strategy.initialize_population(init_key)

        for g in range(num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            fitness, population = strategy.evaluate_population(population, (x0s, ts, ys), eval_key)

            if g < (num_generations-1):
                population = strategy.evolve(population, fitness, sample_key)

        strategy.print_pareto_front(save=True, file_name=f'data/Kozax_results/LV_partial/{seed}')

if __name__ == '__main__':
    evolve_LV()