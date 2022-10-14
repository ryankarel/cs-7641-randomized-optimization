"""UL."""

import time
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

import pickle

RANDOM_STATE = 5403

def sample_edge(bit_string_size):
    return (np.random.randint(0, bit_string_size),
            np.random.randint(0, bit_string_size))

def sample_edges(num_edges, bit_string_size):
    edges = []
    for _ in range(num_edges):
        edges.append(sample_edge(bit_string_size))
    return [(e1, e2) for e1, e2 in edges if e1 != e2]

def convergence_plot(fitness_curve, title, flip=False):
    fitnesses = fitness_curve[:, 0]
    flip_factor = -1 if flip else 1
    plot = pd.Series(flip_factor * fitnesses).plot(
        xlabel='Iterations',
        title=title,
        ylim=(0, None)[::flip_factor],
        ylabel='Fitness'
    )
    return plot

def fevals_plot(fitness_curve, title):
    fevals = fitness_curve[:, 1]
    plot = pd.Series(fevals).plot(
        xlabel='Iterations',
        title=title,
        ylim=(0, None),
        ylabel='# Function Evaluations'
    )
    return plot

def get_new_problems(bit_string_size):
    problems = {
        'Flip Flop': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.FlipFlop()
        ),
        '4-Peaks': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.FourPeaks()
        ),
        '6-Peaks': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.SixPeaks()
        ),
        'Continuous Peaks': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.ContinuousPeaks()
        ),
        'Queens': mlrose.DiscreteOpt(
            length=bit_string_size,
            maximize=False,
            max_val=bit_string_size,
            fitness_fn=mlrose.Queens()
        ),
        'One-Max': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.OneMax()
        ),
        'Max K Color': mlrose.DiscreteOpt(
            length=bit_string_size,
            max_val=int(np.sqrt(bit_string_size)),
            maximize=False,
            fitness_fn=mlrose.MaxKColor(sample_edges(bit_string_size ** 2, bit_string_size))
        )
    }
    return problems

bit_string_size = 16

optimizers = {
    'Simulated Annealing': mlrose.simulated_annealing,
    'MIMIC': mlrose.mimic,
    'Randomized Hill Climbing': mlrose.random_hill_climb,
    'Genetic': mlrose.genetic_alg
}

problem_names = list(get_new_problems(8).keys())

def run_algorithms(bit_string_size, max_attempts=10):
    
    # need to sample these fresh each time to avoid weird feval bug
    problems = get_new_problems(bit_string_size)
    output = {}
    
    for problem_name in problems:
        start = time.time()
        output[problem_name] = {}
        for optim_name in optimizers:
            best_state, best_fitness, fitness_curve = optimizers[optim_name](
                problem=problems[problem_name],
                curve=True,
                random_state=RANDOM_STATE,
                max_attempts=max_attempts,
            )
            output[problem_name][optim_name] = {
                'best_state': best_state,
                'best_fitness': best_fitness,
                'fitness_curve': fitness_curve,
                'wall_clock_time': round((time.time() - start) / 0.01) * 0.01
            }
            
    return output

complete_collection = {}

for bit_str_sz in [8, 16, 32, 64]:
    for mx_atmpts in [10, 50, 100]:
        output = run_algorithms(bit_string_size=bit_str_sz, max_attempts=mx_atmpts)
        complete_collection[f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'] = output
        
pickle.dump(complete_collection, 'complete_collection.pkl')