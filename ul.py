"""UL."""

import time
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from pathlib import Path

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
        # 'Flip Flop': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.FlipFlop()
        # ),
        '4-Peaks': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.FourPeaks()
        ),
        # '6-Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.SixPeaks()
        # ),
        # 'Continuous Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.ContinuousPeaks()
        # ),
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
        # 'Max K Color': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     max_val=int(np.sqrt(bit_string_size)),
        #     maximize=False,
        #     fitness_fn=mlrose.MaxKColor(sample_edges(bit_string_size ** 2, bit_string_size))
        # )
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

negate = {
    'Flip Flop': 1,
    '4-Peaks': 1,
    '6-Peaks': 1,
    'Continuous Peaks': 1,
    'Queens': -1,
    'One-Max': 1,
    'Max K Color': -1
}

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
                random_state=RANDOM_STATE + max_attempts,
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

start = time.time()
for bit_str_sz in [8, 32, 64]:
    for mx_atmpts in [5, 10, 20]:
        key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
        print(key, f'{(time.time() - start) / 60:.1f} m')
        output = run_algorithms(bit_string_size=bit_str_sz, max_attempts=mx_atmpts)
        complete_collection[key] = output
        
with open('complete_collection.pkl', 'wb') as f:
    pickle.dump(str(complete_collection), f)
    
complete_collection.keys()

for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        bit_str_sz = 64
        fitness_curves = pd.DataFrame(columns=['Iteration', 'Fitness', 'Max Attempts'])
        for mx_atmpts in [20, 10, 5]:
            key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
            algos = complete_collection[key][problem_name]
            fc = algos[opt_name]['fitness_curve']
            fitness_curves = fitness_curves.append(
                pd.DataFrame({'Iteration': range(len(fc)),
                              'Fitness': fc[:, 0] * negate[problem_name],
                              'Max Attempts': mx_atmpts})
            )
        plt.figure()
        sns.lineplot(
            data=fitness_curves.reset_index(drop=True),
            x='Iteration',
            y='Fitness',
            hue='Max Attempts'
        ).set_title(f'Convergence Plot - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_convergence.png',
            dpi=400
        )
            
for problem_name in problem_names:
    for opt_name in optimizers:
        mx_atmpts = 10
        fitness_curves = pd.DataFrame(columns=['Iteration', 'Fitness', 'Max Attempts'])
        for bit_str_sz in [8, 32, 64]:
            key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
            algos = complete_collection[key][problem_name]
            fc = algos[opt_name]['fitness_curve']
            fitness_curves = fitness_curves.append(
                pd.DataFrame({'Iteration': range(len(fc)),
                              'Fitness': fc[:, 0] * negate[problem_name],
                              'Input Size': bit_str_sz})
            )
        plt.figure()
        sns.lineplot(
            data=fitness_curves.reset_index(drop=True),
            x='Iteration',
            y='Fitness',
            hue='Input Size'
        ).set_title(f'Problem Size Plot - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_probsize.png',
            dpi=400
        )
            
for problem_name in problem_names:
    comparison = pd.Series()
    for opt_name in optimizers:
        mx_atmpts = 20
        bit_str_sz = 64
        key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
        algos = complete_collection[key][problem_name]
        val = algos[opt_name]['wall_clock_time']
        comparison.loc[opt_name] = val
    comparison.plot(kind='bar', title=f'Wall Clock Time - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/wallclock.png',
        dpi=400
    )
    
for problem_name in problem_names:
    comparison = pd.Series()
    for opt_name in optimizers:
        mx_atmpts = 20
        bit_str_sz = 64
        key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
        algos = complete_collection[key][problem_name]
        val = algos[opt_name]['best_fitness']
        comparison.loc[opt_name] = val
    comparison.plot(kind='bar', title=f'Best Fitness - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/best_fitness.png',
        dpi=400
    )
                
            
    