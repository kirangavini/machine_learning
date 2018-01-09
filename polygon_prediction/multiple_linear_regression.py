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

y = dataset.iloc[:, 7].values
y *=10

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

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






