# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:14:19 2022

@author: rache
"""


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from functools import reduce
import mlrose_hiive as mlrose
from sklearn.base import clone
from copy import deepcopy


random_state = 23523

X_wine = pd.read_pickle(r'X.pkl')
Y_wine = pd.read_pickle(r'Y.pkl')

X_wine_train, X_wine_test, Y_wine_train, Y_wine_test = train_test_split(
    X_wine, Y_wine,
    test_size=0.3,
    random_state=random_state
)


previous_wine_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='logistic',
    max_iter=100,
    learning_rate_init=0.003,
    alpha=0,
    random_state=random_state
)
previous_wine_model.fit(X_wine_train, Y_wine_train)

mlp_classifier = deepcopy(previous_wine_model)

# need to get 1-d array of parameters
starting = {
    'coefs_shapes': [layer.shape for layer in previous_wine_model.coefs_],
    'intercepts_shapes': [layer.shape for layer in previous_wine_model.intercepts_],
    'all_params': 
        reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.coefs_], []) +
        reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.intercepts_], [])
}

def get_total_num_params(starting):
    total_number_coefs = reduce(
        lambda x, y: np.prod(x) + np.prod(y),
        starting['coefs_shapes'],
        (0,)
    )
    total_number_intercepts = reduce(
        lambda x, y: np.prod(x) + np.prod(y),
        starting['intercepts_shapes'],
        (0,)
    )
    return total_number_coefs + total_number_intercepts

def recover_lists_from_all_params(starting):
    total_number_coefs = reduce(
        lambda x, y: np.prod(x) + np.prod(y),
        starting['coefs_shapes'],
        (0,)
    )
    coef_1d = starting['all_params'][:total_number_coefs]
    coef_counter = 0
    coef_list = []
    for shape in starting['coefs_shapes']:
        num_vals = np.prod(shape)
        end = coef_counter + num_vals
        relevant_section = coef_1d[coef_counter:end]
        coefs = np.array(relevant_section).reshape(shape)
        coef_list.append(coefs)
        coef_counter = end
        
    ints_1d = starting['all_params'][total_number_coefs:]
    ints_counter = 0
    ints_list = []
    for shape in starting['intercepts_shapes']:
        num_vals = np.prod(shape)
        end = ints_counter + num_vals
        relevant_section = ints_1d[ints_counter:end]
        intercepts = np.array(relevant_section).reshape(shape)
        ints_list.append(intercepts)
        ints_counter = end
        
    return coef_list, ints_list

def update_model(mlp_classifier, state):
    state_starting = {
        'coefs_shapes': [layer.shape for layer in previous_wine_model.coefs_],
        'intercepts_shapes': [layer.shape for layer in previous_wine_model.intercepts_],
        'all_params': state
    }
    coefs_, intercepts_ = recover_lists_from_all_params(state_starting)
    mlp_classifier.coefs_ = coefs_
    mlp_classifier.intercepts_ = intercepts_
    return mlp_classifier

def score_model(mlp_classifier, X, Y):
    Y_pred = mlp_classifier.predict_proba(X)
    score = roc_auc_score(
        Y,
        Y_pred,
        multi_class="ovo"
    )
    return score
    
def cust_fn(state, mlp_classifier, X_wine_train, Y_wine_train):
    mlp_classifier = update_model(mlp_classifier, state)
    return score_model(mlp_classifier, X_wine_train, Y_wine_train)
    
np.random.seed(random_state)
INIT_NN_STATE = np.random.randn(get_total_num_params(starting)) / 2

kwargs = {
    'mlp_classifier': mlp_classifier,
    'X_wine_train': X_wine_train,
    'Y_wine_train': Y_wine_train
}
custom_fitness = mlrose.CustomFitness(
    cust_fn,
    problem_type='continuous',
    **kwargs
)

def redefine_neural_network_problem():
    continuous_optimization_problem = mlrose.ContinuousOpt(
        length=get_total_num_params(starting),
        fitness_fn=custom_fitness,
        maximize=True,
        min_val=-3,
        max_val=3,
        step=0.10
    )
    return continuous_optimization_problem



# best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(
#     problem=redefine_neural_network_problem(),
#     curve=True,
#     random_state=random_state,
#     init_state=init_state,
#     max_attempts=3
# )

# pd.Series(fitness_curve[:, 0]).plot()

# score_model(
#     update_model(mlp_classifier, best_state),
#     X_wine_test,
#     Y_wine_test
# )

