import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import jax

import diffrax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from kozax.genetic_programming import GeneticProgramming
from pysr import PySRRegressor

class LotkaVolterra():
    def __init__(self):
        self.n_var = 2

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4

    def sample_init_states(self, batch_size, key):
        return jr.uniform(key, shape = (batch_size,2), minval=5, maxval=15)
    
    def drift(self, t, state, args):
        return jnp.array([self.alpha * state[0] - self.beta * state[0] * state[1], self.delta * state[0] * state[1] - self.gamma * state[1]])
    
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

    dy = jax.vmap(lambda ys_: jax.vmap(lambda y: env.drift(0.0,y,None))(ys_))(ys)
    
    return ys, dy

class FitnessFunction:
    def __call__(self, candidate, data, tree_evaluator):
        _X, _Y = data
        pred = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, _X)
        errors = jnp.abs(pred-_Y)/jnp.mean(jnp.abs(_Y))
        fitness = jnp.mean(errors)
        return fitness
    
def test_kozax(seeds, env, args, operator_list, variable_list = ["x0"]):
    population_size, num_populations, num_generations = args

    fitness_function = FitnessFunction()

    layer_sizes = jnp.array([env.n_var])
    strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, num_populations = num_populations,
                                max_nodes = 15, coefficient_optimisation="ES", migration_period=5, size_parsimony=0.003, start_coefficient_optimisation = 0, 
                                optimise_coefficients_elite=100, ES_n_iterations=5, ES_n_offspring = 200, init_learning_rate=0.2)

    for seed in seeds:
        strategy.reset()
        key = jr.PRNGKey(seed)
        print(f"Seed: {seed}")
        key, init_key, data_key = jr.split(key, 3)

        ys, dys = get_data(data_key, env, dt=1.0, T=30, batch_size=1)

        population = strategy.initialize_population(init_key)

        for g in range(num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            fitness, population = strategy.evaluate_population(population, (ys[0], dys[0]), eval_key)

            if g < (num_generations-1):
                
                population = strategy.evolve(population, fitness, sample_key)

        strategy.print_pareto_front(save=True, file_name=f'data/Kozax_results/LV_full/{seed}')

def test_pysr(seeds, env, args):
    population_size, num_populations, num_generations = args

    for seed in seeds:
        key = jr.PRNGKey(seed)
        print(f"Seed: {seed}")
        key, init_key, data_key = jr.split(key, 3)

        ys, dys = get_data(data_key, env, dt=1.0, T=30, batch_size=1)
        
        model = PySRRegressor(
                binary_operators="+ * - / ^".split(" "),
                unary_operators="",
                loss="loss(x, y) = abs(x-y)",
                constraints={"^": (-1, 3)},
                equation_file=f"data/PySR_results/LV_full/{seed}.csv",
                population_size=population_size,
                niterations=num_generations,
                populations=num_populations,
                random_state=seed,
                deterministic=True,
                procs=0,
                multithreading=False,
                verbosity=0
            )

        model.fit(ys[0], dys[0])#, variable_names = variable_list)

def symbolic_regression(seeds, env, method, operator_list, variable_list, args):
    if method=="PySR":
        test_pysr(seeds, env, args)

    elif method=="Kozax":
        test_kozax(seeds, env, args, operator_list, variable_list)

env = LotkaVolterra()

operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                 ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
                 ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                 ]

variable_list = [["x" + str(i) for i in range(env.n_var)]]

seeds = np.arange(10)

args = [100, 10, 50]

method = "Kozax"
# method = "PySR"

symbolic_regression(seeds, env, method, operator_list, variable_list, args)