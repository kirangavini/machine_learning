# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv('enddata.csv')
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

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(metrics.mean_squared_error(y_test, y_pred))

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree Regression')
plt.xlabel('Area')
plt.ylabel('Capacitance')
plt.show()