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
        'Flip Flop': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.FlipFlop()
        ),
        '4-Peaks': mlrose.DiscreteOpt(
            length=bit_string_size,
            fitness_fn=mlrose.FourPeaks()
        ),
        # '6-Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.SixPeaks()
        # ),
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

optimizers = {
    'Simulated Annealing': mlrose.simulated_annealing,
    'MIMIC': mlrose.mimic,
    'Randomized Hill Climbing': mlrose.random_hill_climb,
    'Genetic': mlrose.genetic_alg
}

extra_params = {
    'Simulated Annealing': {
        'GeometricDecay': {'schedule': mlrose.GeomDecay()},
        'ArithmeticDecay': {'schedule': mlrose.ArithDecay()},    
    },
    'MIMIC': {
        'Pop-100,KeepPct-0.2': {'pop_size': 100, 'keep_pct': 0.2},
        'Pop-100,KeepPct-0.1': {'pop_size': 100, 'keep_pct': 0.1},
        'Pop-200,KeepPct-0.2': {'pop_size': 200, 'keep_pct': 0.2},
        'Pop-200,KeepPct-0.1': {'pop_size': 200, 'keep_pct': 0.1},
    },
    'Randomized Hill Climbing': {
        'Restarts-0': {'restarts': 0},
        'Restarts-1': {'restarts': 1},
        'Restarts-2': {'restarts': 2},
        'Restarts-4': {'restarts': 4},
    },
    'Genetic': {
        'Pop-100,MutateProb-0.2': {'pop_size': 100, 'mutation_prob': 0.2},
        'Pop-100,MutateProb-0.1': {'pop_size': 100, 'mutation_prob': 0.1},
        'Pop-200,MutateProb-0.2': {'pop_size': 200, 'mutation_prob': 0.2},
        'Pop-200,MutateProb-0.1': {'pop_size': 200, 'mutation_prob': 0.1},
    }
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

def run_algorithms(bit_string_size=32, max_attempts=10):
    
    # need to sample these fresh each time to avoid weird feval bug
    problems = get_new_problems(bit_string_size)
    output = {}
    start = time.time()
    
    for problem_name in problems:
        start = time.time()
        output[problem_name] = {}
        for optim_name in optimizers:
            extra_param_options = extra_params[optim_name]
            output[problem_name][optim_name] = {}
            for extra_param_key in extra_param_options:
                print(problem_name, optim_name, extra_param_key, f'{(time.time() - start) / 60:.1f}m')
                best_state, best_fitness, fitness_curve = optimizers[optim_name](
                    problem=problems[problem_name],
                    curve=True,
                    random_state=RANDOM_STATE + max_attempts,
                    max_attempts=max_attempts,
                    **extra_param_options[extra_param_key]
                )
                output[problem_name][optim_name][extra_param_key] = {
                    'best_state': best_state,
                    'best_fitness': best_fitness,
                    'fitness_curve': fitness_curve,
                    'wall_clock_time': round((time.time() - start) / 0.01) * 0.01
                }
            
    return output

complete_collection = {}

bit_sz_options = [8, 32, 64]

for bit_str_sz in bit_sz_options:
    complete_collection[f'BS={bit_str_sz}'] = run_algorithms(
        bit_string_size=bit_str_sz,
        max_attempts=10
    )
        
with open('complete_collection.pkl', 'wb') as f:
    pickle.dump(str(complete_collection), f)
    
complete_collection.keys()

for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        bit_str_sz = 32
        fitness_curves = pd.DataFrame(columns=['Iteration', 'Fitness', 'Params'])
        mx_atmpts = 10
        algos = complete_collection[f'BS={bit_str_sz}'][problem_name][opt_name]
        for algo in algos:
            fc = algos[algo]['fitness_curve']
            fitness_curves = fitness_curves.append(
                pd.DataFrame({'Iteration': range(len(fc)),
                              'Fitness': fc[:, 0] * negate[problem_name],
                              'Params': algo})
            )
        plt.figure()
        sns.lineplot(
            data=fitness_curves.reset_index(drop=True),
            x='Iteration',
            y='Fitness',
            hue='Params'
        ).set_title(f'Convergence Plot - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_convergence.png',
            dpi=300
        )
  
for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        bit_str_sz = 32
        fitness_curves = pd.DataFrame(columns=['Iteration', 'F-evals', 'Params'])
        mx_atmpts = 10
        algos = complete_collection[f'BS={bit_str_sz}'][problem_name][opt_name]
        for algo in algos:
            fc = algos[algo]['fitness_curve']
            fitness_curves = fitness_curves.append(
                pd.DataFrame({'Iteration': range(len(fc)),
                              'F-evals': fc[:, 1],
                              'Params': algo})
            )
        plt.figure()
        sns.lineplot(
            data=fitness_curves.reset_index(drop=True),
            x='Iteration',
            y='F-evals',
            hue='Params'
        ).set_title(f'# Function Evaluations - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_fevals.png',
            dpi=300
        )
            
for problem_name in problem_names:
    for opt_name in optimizers:
        mx_atmpts = 10
        fitness_curves = pd.DataFrame(columns=['Iteration', 'Fitness', 'Max Attempts'])
        for bit_str_sz in bit_sz_options:
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
            dpi=300
        )
            
for problem_name in problem_names:
    comparison = pd.Series()
    for opt_name in optimizers:
        mx_atmpts = 10
        bit_str_sz = 64
        key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
        algos = complete_collection[key][problem_name]
        val = algos[opt_name]['wall_clock_time']
        comparison.loc[opt_name] = val
    comparison.plot(kind='bar', title=f'Wall Clock Time - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/wallclock.png',
        dpi=300
    )
    
for problem_name in problem_names:
    comparison = pd.Series()
    for opt_name in optimizers:
        mx_atmpts = 10
        bit_str_sz = 64
        key = f'BitSize={bit_str_sz}, MaxAttempts={mx_atmpts}'
        algos = complete_collection[key][problem_name]
        val = algos[opt_name]['best_fitness'] * negate[problem_name]
        comparison.loc[opt_name] = val
    comparison.plot(kind='bar', title=f'Best Fitness - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/best_fitness.png',
        dpi=300
    )
                
            
    