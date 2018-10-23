# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine

from functools import partial

import numpy as np
import pandas as pd

wine_dataset = load_wine()

# сразу разделим данные на обучающую и тестовую выборки:
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(wine_dataset['data'], wine_dataset['target'], random_state=15)

# axis = 0 указывает, что вычислять нужно по столбцам

u = X_train.mean(axis=0)    # Медиана
o = X_train.std(axis=0)     # Дисперсия

X_train_normalized = (X_train - u) / o

# также есть готовый метод
from sklearn import preprocessing
X_scaled = preprocessing.scale(X_train)

print(X_train_normalized == X_scaled)

M_cov = np.cov(X_train_normalized.T)
# print(M_cov.shape)
# print(M_cov)

eigen_values, eigen_vectors = np.linalg.eig(M_cov)
print("Собственные значение ", eigen_values)
print("Собственные вектора ", eigen_vectors)

# Сортировка
idx = np.argsort(eigen_values)
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:,idx]

# Делаем выборку n собственных векторов с наибольшими значениями
n = 2

B = eigen_vectors[:n]
print("Собственные вектора с макс знач ", B)

# Проекция вектора \ матрицы на матрицу с базисом M
def projection(M, x):
    return ((np.linalg.inv(M @ M.T) @ M) @ x.T).T

Pi = partial(projection, B)

def get_new_data(mu, sigma, project, x):
    return project((x - mu) / sigma)

process = partial(get_new_data, u, o, Pi)

