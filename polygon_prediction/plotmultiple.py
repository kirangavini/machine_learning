import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from matplotlib import cm
from statsmodels.api import add_constant


# collect data into numpy arrays
X = []
Y = []
for line in open('data_2d.csv'): # contains 3 columns: x1, x2, and y
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)]) # here X[i][0] represents x0 = 1
    Y.append(float(y))
X = np.array(X)
Y = np.array(Y)

dataset = pd.read_csv('enddata.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 2].values
one =[]
for i in range(2043):
      one.append(1)
onedata = pd.DataFrame(np.array(one).reshape(2043,1),columns =list(['1'])) 
onedata = onedata.rename(columns={'1':'one'})     
dataset = pd.concat([onedata, dataset], axis=1)       
      
# calculate weights for computing solutions
w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

# calculate r-squared error given weights
Yhat = X.dot(w)
d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared value of", r2)

# calculate plane of best fit
divs = 2 # number of divisions in surface: generates divs^2 points.
         # The surface is a plane, so just 2^2 = 4 points can define it.
# plane spans all values of x1 and x2 from data
x1_range = np.linspace(min(X[:,1]),max(X[:,1]),divs)
x2_range = np.linspace(min(X[:,2]),max(X[:,2]),divs)
X_plane = []
for i in range(divs):
    for j in range(divs):
        X_plane.append([1, x1_range[i], x2_range[j]])
X_plane = np.array(X_plane)
# values of y are equal to the linear regression of points on the plane
Yhat2 = X_plane.dot(w)

# rearrange Yhat2 into a coordinate matrix for display as a surface
Yhat2_surface = []
for i in range(divs):
    Yhat2_surface.append(Yhat2[ divs*i : divs*i+divs ])
Yhat2_surface = np.array(Yhat2_surface)
Yhat2 = Yhat2_surface

# generate coordinate matrices for x1 and x2 values
X2, X1 = np.meshgrid(x1_range, x2_range) # intentional ordering: X2, *then* X1

# plot results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,2], Y) # supplied data
ax.set_xlabel('No of Points') # X axis data label
ax.set_ylabel('Area') # Y axis data label
ax.set_zlabel('capacitance') # Z axis data label

plt.show()
