"""UL."""

import time
import mlrose_hiive as mlrose
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from pathlib import Path

from nn import redefine_neural_network_problem

RANDOM_STATE = 5403

def sample_edge(bit_string_size):
    return (np.random.randint(0, bit_string_size),
            np.random.randint(0, bit_string_size))

def sample_edges(num_edges, bit_string_size):
    edges = []
    for _ in range(num_edges):
        edges.append(sample_edge(bit_string_size))
    return [(e1, e2) for e1, e2 in edges if e1 != e2]

def get_new_problems(bit_string_size):
    problems = {
        # 'Flip Flop': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.FlipFlop()
        # ),
        # '4-Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.FourPeaks()
        # ),
        # '6-Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.SixPeaks()
        # ),
        # 'Continuous Peaks': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.ContinuousPeaks()
        # ),
        # 'Queens': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     maximize=False,
        #     max_val=bit_string_size,
        #     fitness_fn=mlrose.Queens()
        # ),
        # 'One-Max': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     fitness_fn=mlrose.OneMax()
        # ),
        # 'Max K Color': mlrose.DiscreteOpt(
        #     length=bit_string_size,
        #     max_val=int(np.sqrt(bit_string_size)),
        #     maximize=False,
        #     fitness_fn=mlrose.MaxKColor(sample_edges(bit_string_size ** 2, bit_string_size))
        # ),
        'Wine Neural Network': redefine_neural_network_problem()
    }
    return problems

optimizers = {
    'Simulated Annealing': mlrose.simulated_annealing,
    # 'MIMIC': mlrose.mimic,
    'Randomized Hill Climbing': mlrose.random_hill_climb,
    'Genetic': mlrose.genetic_alg
}

extra_params = {
    'Simulated Annealing': {
        'GeometricDecay-0.8': {'schedule': mlrose.GeomDecay(decay=0.8)},
        'ArithmeticDecay-0.01': {'schedule': mlrose.ArithDecay(decay=0.01)}, 
        'GeometricDecay-0.95': {'schedule': mlrose.GeomDecay(decay=0.95)},
        'ArithmeticDecay-0.03': {'schedule': mlrose.ArithDecay(decay=0.03)},    
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

problem_names = list(get_new_problems(8).keys())#[:-1]

negate = {
    'Flip Flop': 1,
    '4-Peaks': 1,
    '6-Peaks': 1,
    'Continuous Peaks': 1,
    'Queens': -1,
    'One-Max': 1,
    'Max K Color': -1,
    'Wine Neural Network': 1
}

def run_algorithms(bit_string_size=16, max_attempts=8):
    
    # need to sample these fresh each time to avoid weird feval bug
    problems = get_new_problems(bit_string_size)
    output = {}
    start = time.time()
    
    for problem_name in problems:
        output[problem_name] = {}
        for optim_name in optimizers:
            extra_param_options = extra_params[optim_name]
            output[problem_name][optim_name] = {}
            for extra_param_key in extra_param_options:
                print(problem_name, optim_name, extra_param_key, f'{(time.time() - start) / 60:.1f}m')
                if problem_name == 'Wine Neural Network':
                    if bit_string_size != 16 or optim_name == 'MIMIC':
                        continue
                    best_state, best_fitness, fitness_curve = optimizers[optim_name](
                        problem=get_new_problems(bit_string_size)[problem_name],
                        curve=True,
                        random_state=RANDOM_STATE + max_attempts,
                        max_attempts=5,
                        max_iters=1000,
                        **extra_param_options[extra_param_key]
                    )
                else:
                    best_state, best_fitness, fitness_curve = optimizers[optim_name](
                        problem=get_new_problems(bit_string_size)[problem_name],
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

bit_sz_options = [8, 16, 32]

for bit_str_sz in bit_sz_options:
    complete_collection[f'BS={bit_str_sz}'] = run_algorithms(
        bit_string_size=bit_str_sz,
        max_attempts=12
    )
        
with open('complete_collection.pkl', 'wb') as f:
    pickle.dump(str(complete_collection), f)
    
complete_collection.keys()

# convergences
for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        bit_str_sz = 16
        fitness_curves = pd.DataFrame(columns=['Iteration', 'Fitness', 'Params'])
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
  
# f-evals
for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        bit_str_sz = 16
        fitness_curves = pd.DataFrame(columns=['Iteration', 'F-evals', 'Params'])
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
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    for opt_name in optimizers:
        fitness_curves = pd.DataFrame(columns=['Iteration', 'F-evals', 'Params'])
        for bit_str_sz in bit_sz_options:
            algos = complete_collection[f'BS={bit_str_sz}'][problem_name][opt_name]
            for algo in algos:
                fc = algos[algo]['fitness_curve']
                fitness_curves = fitness_curves.append(
                    pd.DataFrame({'Wall Clock Time': algos[algo]['wall_clock_time'],
                                  'F-evals': fc[:, 1].max(),
                                  'Best Fitness': algos[algo]['best_fitness'] * negate[problem_name],
                                  'Params': algo,
                                  'Input Size': bit_str_sz}, index=[0])
                )
        plt.figure()
        sns.barplot(
            data=fitness_curves.reset_index(drop=True),
            x='Input Size',
            y='F-evals',
            hue='Params'
        ).set_title(f'Input Size Impact on F-evals - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_inputsize_fevals.png',
            dpi=300
        )
            
        plt.figure()
        sns.barplot(
            data=fitness_curves.reset_index(drop=True),
            x='Input Size',
            y='Wall Clock Time',
            hue='Params'
        ).set_title(f'Input Size Impact on Wall Clock Time - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_inputsize_wallclock.png',
            dpi=300
        )
            
        plt.figure()
        sns.barplot(
            data=fitness_curves.reset_index(drop=True),
            x='Input Size',
            y='Best Fitness',
            hue='Params'
        ).set_title(f'Best Fitness - {problem_name} - {opt_name}').get_figure().savefig(
            f'plots/{problem_name}/{opt_name.replace(" ", "")}_bestfitness.png',
            dpi=300
        )
            

for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    fitness_curves = pd.DataFrame()
    for opt_name in optimizers:
        bit_str_sz = 16
        algos = complete_collection[f'BS={bit_str_sz}'][problem_name][opt_name]
        best_fitness = max(
            algos[algo]['best_fitness'] * negate[problem_name]
            for algo in algos
        )
        fitness_curves = fitness_curves.append(
            pd.DataFrame({'Best Fitness': best_fitness,
                          'Algorithm': opt_name}, index=[0])
        )
            
    plt.figure()
    sns.barplot(
        data=fitness_curves.reset_index(drop=True),
        x='Algorithm',
        y='Best Fitness'
    ).set_title(f'Best Fitness across Optimizers - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/best_ult_fitness.png',
        dpi=300
    )
        
for problem_name in problem_names:
    Path(f'plots/{problem_name}').mkdir(parents=True, exist_ok=True)
    fitness_curves = pd.DataFrame()
    for opt_name in optimizers:
        bit_str_sz = 32
        algos = complete_collection[f'BS={bit_str_sz}'][problem_name][opt_name]
        best_fitness = np.mean([
            algos[algo]['wall_clock_time']
            for algo in algos
        ])
        fitness_curves = fitness_curves.append(
            pd.DataFrame({'Average Time': best_fitness,
                          'Algorithm': opt_name}, index=[0])
        )
            
    plt.figure()
    sns.barplot(
        data=fitness_curves.reset_index(drop=True),
        x='Algorithm',
        y='Average Time'
    ).set_title(f'Average Time across Optimizers - {problem_name}').get_figure().savefig(
        f'plots/{problem_name}/avg_time.png',
        dpi=300
    )
            

         