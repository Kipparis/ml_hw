# -*- coding: utf-8 -*-

from sklearn.datasets import load_wine

from functools import partial

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

cm = plt.get_cmap("cool")

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


# Применим полученные знания
X_train_processed = process(X_train)
X_test_processed = process(X_test)

print(X_train_processed.shape, "vs", X_train.shape)
print(X_test_processed.shape, "vs", X_test.shape)

logreg = LogisticRegression(C=1e4, solver='lbfgs', multi_class='multinomial', max_iter=10000)
result = logreg.fit(X_train_processed, y_train)

print("Точность на тестовом наборе: {:.2f}".format(logreg.score(X_test_processed, y_test)))



X = X_test_processed
Y = y_test
print("\nX shape ", X.shape, '\ny shape ', Y.shape)

shift = 0.2
x_min, x_max = X[:, 0].min() - shift, X[:, 0].max() + shift
y_min, y_max = X[:, 1].min() - shift, X[:, 1].max() + shift
h = .01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=cm)

plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=cm)

plt.show()