"""UL."""

import time
import mlrose_hiive as mlrose
import numpy as np

BIT_STRING_SIZE = 16
RANDOM_STATE = 5403

def sample_edge():
    return (np.random.randint(0, BIT_STRING_SIZE), np.random.randint(0, BIT_STRING_SIZE))

def sample_edges(num_edges):
    edges = []
    for _ in range(num_edges):
        edges.append(sample_edge())
    return [(e1, e2) for e1, e2 in edges if e1 != e2]


problems = {
    'Flip Flop': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        fitness_fn=mlrose.FlipFlop()
    ),
    '4-Peaks': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        fitness_fn=mlrose.FourPeaks()
    ),
    '6-Peaks': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        fitness_fn=mlrose.SixPeaks()
    ),
    'Continuous Peaks': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        fitness_fn=mlrose.ContinuousPeaks()
    ),
    'Queens': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        maximize=False,
        max_val=BIT_STRING_SIZE,
        fitness_fn=mlrose.Queens()
    ),
    'One-Max': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        fitness_fn=mlrose.OneMax()
    ),
    'Max K Color': mlrose.DiscreteOpt(
        length=BIT_STRING_SIZE,
        max_val=int(np.sqrt(BIT_STRING_SIZE)),
        maximize=False,
        fitness_fn=mlrose.MaxKColor(sample_edges(BIT_STRING_SIZE ** 2))
    )
}

initial_state = np.zeros((BIT_STRING_SIZE,))

optimizers = {
    'Simulated Annealing': mlrose.simulated_annealing,
    'MIMIC': mlrose.mimic,
    'Randomized Hill Climbing': mlrose.random_hill_climb,
    'Genetic': mlrose.genetic_alg
}

output = {}

for problem_name in problems:
    start = time.time()
    output[problem_name] = {}
    for optim_name in optimizers:
        best_state, best_fitness, fitness_curve = optimizers[optim_name](
            problem=problems[problem_name],
            max_iters=500,
            #init_state=initial_state,
            curve=True,
            random_state=RANDOM_STATE
        )
        output[problem_name][optim_name] = {
            'best_state': best_state,
            'best_fitness': best_fitness,
            'fitness_curve': fitness_curve,
            'wall_clock_time': round((time.time() - start) / 0.01) * 0.01
        }

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
