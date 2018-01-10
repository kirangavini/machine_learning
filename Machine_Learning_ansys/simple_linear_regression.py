# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

# Importing the dataset
#dataset = pd.read_csv('enddata.csv')
dataset = pd.read_csv('enddata_angside.csv')
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
X = dataset.iloc[:, 6:7].values

vararea = np.var(dataset.iloc[:, 1:2].values)
varmaxangle = np.var(dataset.iloc[:,3:4].values)
varminangle = np.var(dataset.iloc[:,4:5].values)
varmaxside = np.var(dataset.iloc[:,5:6].values)
varminside = np.var(dataset.iloc[:,6:7].values)




X= X/max(X)
y = dataset.iloc[:, 2].values 

y *=10

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)

#
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


y_abs = (abs(y_test-y_pred)/y_test)*100

y_abs_min = min(y_abs)
y_abs_max = max(y_abs)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print(metrics.mean_squared_error(y_test, y_pred))



# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Area vs Capacitance(training set)')
plt.xlabel('area')
plt.ylabel('capacitance')
plt.show()
#
## Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Area vs Capacitance(test set)')
plt.xlabel('area')
plt.ylabel('capacitance')
plt.show()