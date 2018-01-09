# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:06:16 2017

@author: kgavini
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import math


point8 = pd.read_csv('8point.csv')

max_angle= []
min_angle= []
angle= []
max_length = []
min_length = []
def minlength8(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)


for i in range(len(point8)):
   angle1 = angles(point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i],point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i])
   angle2 = angles(point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i],point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i])
   angle3 = angles(point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i],point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i])
   angle4 = angles(point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i],point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i])
   angle5 = angles(point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i],point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i])
   angle6 = angles(point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i],point8['x7'][i],point8['y7'][i],point8['x8'][i],point8['y8'][i])
   angle7 = angles(point8['x7'][i],point8['y7'][i],point8['x8'][i],point8['y8'][i],point8['x8'][i],point8['y8'][i],point8['x1'][i],point8['y1'][i])
   angle8 = angles(point8['x8'][i],point8['y8'][i],point8['x1'][i],point8['y1'][i],point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8))
   max_length.append(maxlength8(point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i],point8['x8'],point8['y8']))
   min_length.append(minlength8(point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i],point8['x8'],point8['y8'])) 