# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from matplotlib import cm
from statsmodels.api import add_constant


dataset = pd.read_csv('endresult.csv')
dataset = pd.read_csv('isosclesdata1.csv')
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset = dataset.ix[1:7500,:]
one =[]
for i in range(10000):
      one.append(1)
onedata = pd.DataFrame(np.array(one).reshape(10000,1),columns =list(['1'])) 
onedata = onedata.rename(columns={'1':'one'})     
dataset = pd.concat([onedata, dataset], axis=1)   

dataset['area'] = dataset['area']/max(dataset['area'])
dataset['perimeter_new'] = dataset['perimeter_new']/max(dataset['perimeter_new'])
dataset['x1'] = dataset['x1']/max(dataset['x1'])
dataset['y1'] = dataset['y1']/max(dataset['y1'])
dataset['x2'] = dataset['x2']/max(dataset['x2'])
dataset['y2'] = dataset['y2']/max(dataset['y2'])
dataset['x3'] = dataset['x3']/max(dataset['x3'])
dataset['y3'] = dataset['y3']/max(dataset['y3'])




dataset['No of points'] = dataset['No of points']/max(dataset['No of points'])
dataset['max_angle'] = dataset['max_angle']/max(dataset['max_angle'])
dataset['min_angle'] = dataset['min_angle']/max(dataset['min_angle'])
dataset['max_length'] = dataset['max_length']/max(dataset['max_length'])
dataset['min_length'] = dataset['min_length']/max(dataset['min_length'])



# Importing the dataset

X = dataset.iloc[:, 0:21].values

y = dataset.iloc[:, 21].values


y *=10

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X_train = X_train[1:1500,:]
y_train = y_train[1:1500]






# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
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


fig, ax = plt.subplots()
scat = ax.scatter(dataset['area'], y, c=dataset['No of points'], s=1, marker='o')
fig.colorbar(scat)
plt.xlabel('area')
plt.ylabel('capacitance')


plt.show()






