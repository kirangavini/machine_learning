# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:19:48 2017

@author: kgavini
"""

# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
# Importing the dataset

#dataset = pd.read_csv('enddata.csv')
dataset = pd.read_csv('enddata_angside.csv')
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

one =[]
for i in range(2043):
      one.append(1)
onedata = pd.DataFrame(np.array(one).reshape(2043,1),columns =list(['1'])) 
onedata = onedata.rename(columns={'1':'one'})     
dataset = pd.concat([onedata, dataset], axis=1)   

dataset['area'] = dataset['area']/max(dataset['area'])
dataset['No of points'] = dataset['No of points']/max(dataset['No of points'])

# Importing the dataset

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
y *=10

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)


ypred_linear = lin_reg.predict(X_test)
ypred_polynomial = lin_reg_2.predict(poly_reg.fit_transform(X_test))


y_abs = (abs(y_test-ypred_polynomial)/y_test)*100

y_abs_min = min(y_abs)
y_abs_max = max(y_abs)


print(np.sqrt(metrics.mean_squared_error(y_test, ypred_polynomial)))

print(metrics.mean_squared_error(y_test, ypred_polynomial))

# Visualising the Polynomial Regression results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_train, color = 'blue')
plt.title('Polynomial Regression-(Train data)')
plt.xlabel('Area')
plt.ylabel('Capacitance')
plt.show()

plt.scatter(X_test, ypred_polynomial, color = 'red')
plt.plot(X_test, ypred_polynomial, color = 'blue')
plt.title('Polynomial Regression-(Validation data)')
plt.xlabel('Area')
plt.ylabel('Capacitance')
plt.show()




# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression- Uniform Step size')
plt.xlabel('Area')
plt.ylabel('Capacitance')
plt.show()







