# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:37:43 2017

@author: kgavini
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from keras import optimizers
# load dataset
dataset = pd.read_csv('enddata.csv')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = Sequential()
model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
#model.add(Dense(10, input_dim=50, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.summary()
optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.7)
#optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='mean_squared_error', optimizer='adam')

regressor = model.fit(X_train, y_train, batch_size=50, epochs=500)
	
predictions = model.predict(X_test)

y_abs = np.mean((abs(y_test-predictions)/y_test))

print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(metrics.mean_squared_error(y_test, predictions))

