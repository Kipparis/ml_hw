import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

cm = plt.get_cmap("cool")

# генерируем набор данных
reg = linear_model.Lasso(alpha=0.1)
X = [[1.3],[2.2],[3.1],[4],[6],[8]]
y = [10,9,12,14,16,13]
reg.fit(X, y)

regr = LinearRegression()

fig = plt.figure(figsize=(15, 15))
for d in range(0, 20):  # Степен полиномиальной регрессии
    xc = PolynomialFeatures(degree=d).fit_transform(X)
    regr = regr.fit(xc, y)
    yr = regr.predict(xc)
    plt.plot(X, yr, label='polynomial (d=%i)'%d, lw=2, linestyle='-')
    xc = PolynomialFeatures(degree=d).fit_transform(X)
    print('Полином {0} степени.\tОценка дисперсии: {1}'.format(d, regr.score(xc, y)))

    # рисуем результаты

plt.scatter(X, y, label="data", lw=3, color='red')
plt.legend(loc='upper left')

plt.show()