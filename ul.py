"""UL."""

import time
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd

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

problem_names = [
    'Flip Flop',
    '4-Peaks',
    '6-Peaks',
    'Continuous Peaks',
    'Queens',
    'One-Max',
    'Max K Color'
]

def run_algorithms(bit_string_size, extra_params=None):
    
    # need to sample these fresh each time to avoid weird feval bug
    problems = get_new_problems(bit_string_size)
    output = {}
    
    for problem_name in problems:
        start = time.time()
        output[problem_name] = {}
        for optim_name in optimizers:
            selected_extra_params = {}
            best_state, best_fitness, fitness_curve = optimizers[optim_name](
                problem=problems[problem_name],
                max_iters=50,
                curve=True,
                random_state=RANDOM_STATE,
                max_attempts=100
            )
            output[problem_name][optim_name] = {
                'best_state': best_state,
                'best_fitness': best_fitness,
                'fitness_curve': fitness_curve,
                'wall_clock_time': round((time.time() - start) / 0.01) * 0.01
            }
            
    return output

run_algorithms(8)

print('\nbest fitness')
for problem_name in problems:
    best_fitni = {
         key: output[problem_name][key]['best_fitness']
         for key in output[problem_name]
    }
    print(problem_name, best_fitni)

print('\nwall clock time')
for problem_name in problems:
    best_fitni = {
         key: output[problem_name][key]['wall_clock_time']
         for key in output[problem_name]
    }
    print(problem_name, best_fitni)
    
output['Flip Flop']['MIMIC']
