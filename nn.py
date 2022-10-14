# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:14:19 2022

@author: rache
"""


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

random_state = 23523

previous_wine_model = MLPClassifier(
    hidden_layer_sizes=(50,),
    activation='logistic',
    max_iter=100,
    alpha=0,
    random_state=random_state
)

X_wine = pd.read_pickle(r'C:\Users\rache\cs-7641-supervised-learning\data\wine\X.pkl')
Y_wine = pd.read_pickle(r'C:\Users\rache\cs-7641-supervised-learning\data\wine\Y.pkl')

X_wine_train, X_wine_test, Y_wine_train, Y_wine_test = train_test_split(
    X_wine, Y_wine,
    test_size=0.3,
    random_state=random_state
)

previous_wine_model.fit(X_wine_train, Y_wine_train)

previous_wine_model.coefs_