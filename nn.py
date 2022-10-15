# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:14:19 2022

@author: rache
"""


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from functools import reduce

random_state = 23523

X_wine = pd.read_pickle(r'C:\Users\rache\cs-7641-supervised-learning\data\wine\X.pkl')
Y_wine = pd.read_pickle(r'C:\Users\rache\cs-7641-supervised-learning\data\wine\Y.pkl')

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

# need to get 1-d array of parameters
starting = {
    'coefs_shapes': [layer.shape for layer in previous_wine_model.coefs_],
    'coefs_values': reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.coefs_], []),
    'intercepts_shapes': [layer.shape for layer in previous_wine_model.intercepts_],
    'intercepts_values': reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.intercepts_], []),
    'all_params': 
        reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.coefs_], []) +
        reduce(lambda x, y: x + y, [list(layer.ravel()) for layer in previous_wine_model.intercepts_], [])
}

def recover_lists_from_all_params(all_params)
ending = starting

previous_wine_model.coefs_[0].shape

orig = previous_wine_model.predict_proba(X_wine_test.head(3))

previous_wine_model.intercepts_[1] = np.array([0.5, -0.2, -0.2])
new = previous_wine_model.predict_proba(X_wine_test.head(3))
new - orig
