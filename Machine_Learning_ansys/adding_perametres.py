# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:52:21 2017

@author: kgavini
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import math
dataset = pd.read_csv('endresult.csv')

dataset1 = pd.read_csv('iso.csv')




mean_max_angle = np.mean(dataset['max_angle'])
mean_min_angle = np.mean(dataset['min_angle'])
mean_max_side =  np.mean(dataset['max_length'])
mean_min_side =  np.mean(dataset['min_length'])


var_min_angle =  np.var(dataset['min_angle'])

std_min_angle = np.std(dataset['min_angle'])
std_min_side =  np.std(dataset['min_length'])




mean_maxangle = []
mean_minangle = []
mean_mxside = []
mean_mnside = []
var_mnangle = []
std_mnangle = []
std_mnside =[]
maxcos_val =[]
mincos_val = []
 


for i in range(len(dataset['max_angle'])):
    if dataset['max_angle'][i] > mean_max_angle:
        mean_maxangle.append(1)
    else: mean_maxangle.append(0)    
    
    if dataset['min_angle'][i] > mean_min_angle:
        mean_minangle.append(1)
    else: mean_minangle.append(0) 
    
    if dataset['max_length'][i] > mean_max_side:
        mean_mxside.append(1)
    else: mean_mxside.append(0) 
    
    if dataset['min_length'][i] > mean_min_side:
        mean_mnside.append(1)
    else: mean_mnside.append(0)
    

   
    if dataset['min_length'][i] > var_min_angle:
        var_mnangle.append(1)
    else: var_mnangle.append(0)  
    
    if dataset['min_angle'][i] > std_min_angle:
        std_mnangle.append(1)
    else: std_mnangle.append(0) 
      
    if dataset['min_length'][i] > std_min_side:
        std_mnside.append(1)
    else: std_mnside.append(0)
    
    maxcos_val.append(math.cos(dataset['max_angle'][i]))
    
    mincos_val.append(math.cos(dataset['min_angle'][i]))
    
        
    
dataset1 = pd.DataFrame({'mean_maxangle': mean_maxangle,'mean_minangle': mean_minangle,'mean_mxside': mean_mxside,'mean_mnside': mean_mnside,'var_mnangle': var_mnangle,'std_mnangle': std_mnangle,'std_mnside': std_mnside,'maxcos_val': maxcos_val,'mincos_val': mincos_val})    

dataset = pd.concat([dataset, dataset1], axis=1)

dataset =dataset[['x1','y1','x2','y2','x3','y3','area', 'max_angle', 'min_angle', 'max_length',
       'min_length', 'maxcos_val', 'mean_maxangle',
       'mean_minangle', 'mean_mnside', 'mean_mxside', 'mincos_val',
       'std_mnangle', 'std_mnside', 'var_mnangle','capacitance']]



def capexp(A,L):
    eps = 4.4*8.8541*math.pow(10,-12)
    d = 5*math.pow(10,-3)
    A= A*math.pow(10,-6)
    L= L*math.pow(10,-3) 
    Cf = 8.93*math.pow(10,-12)
    cap = (((eps*A)/d) + Cf*L)/math.pow(10,-12)
    return cap


capacitance_cal= []
for i in range(len(dataset['area'])):
    capacitance_cal.append(capexp(dataset['area'][i],dataset['perimeter_new'][i]))
    
capdata = pd.DataFrame({'cap_exp':capacitance_cal})    
dataset = pd.concat([dataset, capdata], axis=1)    


val = dataset['capacitance']-dataset['cap_exp']
    
y_abs = (abs(val)/dataset['capacitance'])*100

y_abs_min = min(y_abs)
y_abs_max = max(y_abs)    
    
for i in range(len(dataset['area'])):
      if (dataset['capacitance'][i] == 0.69668999999999992):
          print(i)


    
dataset.to_csv('endresult.csv', sep=',')    
      

dataset.to_csv('isosclesdata1.csv', sep=',')
    
      
    
    