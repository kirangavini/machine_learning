# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:31:49 2017

@author: kgavini
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import math

point3 = pd.read_csv('3point.csv')
point4 = pd.read_csv('4point.csv')
point5 = pd.read_csv('5point.csv')
point6 = pd.read_csv('6point.csv')
point7 = pd.read_csv('7point.csv')
point8 = pd.read_csv('8point.csv')
point9 = pd.read_csv('9point.csv')
point10 = pd.read_csv('10point.csv')
dataset = pd.read_csv('enddata.csv')

def angles(x1,y1,x2,y2,x3,y3,x4,y4):
   m1 = y2-y1
   m2 = y4-y3
    
   ang1 = math.atan(m1)
   ang2 = math.atan(m2)
#   angle = 180-abs(ang1-ang2)
   angle = math.degrees(math.pi -abs(ang1-ang2))
    
   return angle

def maxside(x1,y1,x2,y2,x3,y3):
    side1 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    side2 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    side3 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    maxside = max(side1,side2,side3)
    return maxside
    
 
def minside(x1,y1,x2,y2,x3,y3):
    side1 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    side2 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    side3 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    minside = min(side1,side2,side3)
    return minside

def maxlength4(x1,y1,x2,y2,x3,y3,x4,y4):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)



def minlength4(x1,y1,x2,y2,x3,y3,x4,y4):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))
    return min(side)

def maxlength5(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)


def minlength5(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)


def maxlength6(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)

def minlength6(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)

def maxlength7(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)

def minlength7(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)

def maxlength8(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)

def minlength8(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)



def maxlength9(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)

def minlength9(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)

def maxlength10(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9],[x10,y10]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return max(side)


def minlength10(x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10):
    val = [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6],[x7,y7],[x8,y8],[x9,y9],[x10,y10]]
    side = []
    for i in range(len(val)):
      for j in range(i+1,len(val)):
         side.append(np.sqrt((val[j][0]-val[i][0])**2+(val[j][1]-val[i][1])**2))         
    return min(side)

max_angle= []
min_angle= []
angle= []
max_length = []
min_length = []


for i in range(len(point3)):
    
  angle1 = angles(point3['x1'][i],point3['y1'][i],point3['x2'][i],point3['y2'][i],point3['x2'][i],point3['y2'][i],point3['x3'][i],point3['y3'][i])
  angle2 = angles(point3['x2'][i],point3['y2'][i],point3['x3'][i],point3['y3'][i],point3['x3'][i],point3['y3'][i],point3['x1'][i],point3['y1'][i])
  angle3 = angles(point3['x3'][i],point3['y3'][i],point3['x1'][i],point3['y1'][i],point3['x1'][i],point3['y1'][i],point3['x2'][i],point3['y2'][i])
  max_angle.append(max(angle1,angle2,angle3))
  min_angle.append(min(angle1,angle2,angle3))
  max_length.append(maxside(point3['x1'][i],point3['y1'][i],point3['x2'][i],point3['y2'][i],point3['x3'][i],point3['y3'][i]))
  min_length.append(minside(point3['x1'][i],point3['y1'][i],point3['x2'][i],point3['y2'][i],point3['x3'][i],point3['y3'][i]))



for i in range(len(point4)):
   angle1 = angles(point4['x1'][i],point4['y1'][i],point4['x2'][i],point4['y2'][i],point4['x2'][i],point4['y2'][i],point4['x3'][i],point4['y3'][i])
   angle2 = angles(point4['x2'][i],point4['y2'][i],point4['x3'][i],point4['y3'][i],point4['x3'][i],point4['y3'][i],point4['x4'][i],point4['y4'][i])
   angle3 = angles(point4['x3'][i],point4['y3'][i],point4['x4'][i],point4['y4'][i],point4['x4'][i],point4['y4'][i],point4['x1'][i],point4['y1'][i])
   angle4 = angles(point4['x4'][i],point4['y4'][i],point4['x1'][i],point4['y1'][i],point4['x1'][i],point4['y1'][i],point4['x2'][i],point4['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4))
   min_angle.append(min(angle1,angle2,angle3,angle4))
   max_length.append(maxlength4(point4['x1'][i],point4['y1'][i],point4['x2'][i],point4['y2'][i],point4['x3'][i],point4['y3'][i],point4['x4'][i],point4['y4'][i]))
   min_length.append(minlength4(point4['x1'][i],point4['y1'][i],point4['x2'][i],point4['y2'][i],point4['x3'][i],point4['y3'][i],point4['x4'][i],point4['y4'][i]))
   
   
for i in range(len(point5)):
   angle1 = angles(point5['x1'][i],point5['y1'][i],point5['x2'][i],point5['y2'][i],point5['x2'][i],point5['y2'][i],point5['x3'][i],point5['y3'][i])
   angle2 = angles(point5['x2'][i],point5['y2'][i],point5['x3'][i],point5['y3'][i],point5['x3'][i],point5['y3'][i],point5['x4'][i],point5['y4'][i])
   angle3 = angles(point5['x3'][i],point5['y3'][i],point5['x4'][i],point5['y4'][i],point5['x4'][i],point5['y4'][i],point5['x5'][i],point5['y5'][i])
   angle4 = angles(point5['x4'][i],point5['y4'][i],point5['x5'][i],point5['y5'][i],point5['x5'][i],point5['y5'][i],point5['x1'][i],point5['y1'][i])
   angle5 = angles(point5['x5'][i],point5['y5'][i],point5['x1'][i],point5['y1'][i],point5['x1'][i],point5['y1'][i],point5['x2'][i],point5['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5))
   max_length.append(maxlength5(point5['x1'][i],point5['y1'][i],point5['x2'][i],point5['y2'][i],point5['x3'][i],point5['y3'][i],point5['x4'][i],point5['y4'][i],point5['x5'][i],point5['y5'][i]))
   min_length.append(minlength5(point5['x1'][i],point5['y1'][i],point5['x2'][i],point5['y2'][i],point5['x3'][i],point5['y3'][i],point5['x4'][i],point5['y4'][i],point5['x5'][i],point5['y5'][i]))   
   
   
for i in range(len(point6)):
   angle1 = angles(point6['x1'][i],point6['y1'][i],point6['x2'][i],point6['y2'][i],point6['x2'][i],point6['y2'][i],point6['x3'][i],point6['y3'][i])
   angle2 = angles(point6['x2'][i],point6['y2'][i],point6['x3'][i],point6['y3'][i],point6['x3'][i],point6['y3'][i],point6['x4'][i],point6['y4'][i])
   angle3 = angles(point6['x3'][i],point6['y3'][i],point6['x4'][i],point6['y4'][i],point6['x4'][i],point6['y4'][i],point6['x5'][i],point6['y5'][i])
   angle4 = angles(point6['x4'][i],point6['y4'][i],point6['x5'][i],point6['y5'][i],point6['x5'][i],point6['y5'][i],point6['x6'][i],point6['y6'][i])
   angle5 = angles(point6['x5'][i],point6['y5'][i],point6['x6'][i],point6['y6'][i],point6['x6'][i],point6['y6'][i],point6['x1'][i],point6['y1'][i])
   angle6 = angles(point6['x6'][i],point6['y6'][i],point6['x1'][i],point6['y1'][i],point6['x1'][i],point6['y1'][i],point6['x2'][i],point6['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5,angle6))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5,angle6))
   max_length.append(maxlength6(point6['x1'][i],point6['y1'][i],point6['x2'][i],point6['y2'][i],point6['x3'][i],point6['y3'][i],point6['x4'][i],point6['y4'][i],point6['x5'][i],point6['y5'][i],point6['x6'][i],point6['y6'][i]))
   min_length.append(minlength6(point6['x1'][i],point6['y1'][i],point6['x2'][i],point6['y2'][i],point6['x3'][i],point6['y3'][i],point6['x4'][i],point6['y4'][i],point6['x5'][i],point6['y5'][i],point6['x6'][i],point6['y6'][i]))  

for i in range(len(point7)):
   angle1 = angles(point7['x1'][i],point7['y1'][i],point7['x2'][i],point7['y2'][i],point7['x2'][i],point7['y2'][i],point7['x3'][i],point7['y3'][i])
   angle2 = angles(point7['x2'][i],point7['y2'][i],point7['x3'][i],point7['y3'][i],point7['x3'][i],point7['y3'][i],point7['x4'][i],point7['y4'][i])
   angle3 = angles(point7['x3'][i],point7['y3'][i],point7['x4'][i],point7['y4'][i],point7['x4'][i],point7['y4'][i],point7['x5'][i],point7['y5'][i])
   angle4 = angles(point7['x4'][i],point7['y4'][i],point7['x5'][i],point7['y5'][i],point7['x5'][i],point7['y5'][i],point7['x6'][i],point7['y7'][i])
   angle5 = angles(point7['x5'][i],point7['y5'][i],point7['x6'][i],point7['y6'][i],point7['x6'][i],point7['y6'][i],point7['x7'][i],point7['y7'][i])
   angle6 = angles(point7['x6'][i],point7['y6'][i],point7['x7'][i],point7['y7'][i],point7['x7'][i],point7['y7'][i],point7['x1'][i],point7['y1'][i])
   angle7 = angles(point7['x7'][i],point7['y7'][i],point7['x1'][i],point7['y1'][i],point7['x1'][i],point7['y1'][i],point7['x2'][i],point7['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5,angle6,angle7))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5,angle6,angle7))
   max_length.append(maxlength7(point7['x1'][i],point7['y1'][i],point7['x2'][i],point7['y2'][i],point7['x3'][i],point7['y3'][i],point7['x4'][i],point7['y4'][i],point7['x5'][i],point7['y5'][i],point7['x6'][i],point7['y6'][i],point7['x7'][i],point7['y7'][i]))
   min_length.append(minlength7(point7['x1'][i],point7['y1'][i],point7['x2'][i],point7['y2'][i],point7['x3'][i],point7['y3'][i],point7['x4'][i],point7['y4'][i],point7['x5'][i],point7['y5'][i],point7['x6'][i],point7['y6'][i],point7['x7'][i],point7['y7'][i])) 
   
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
   max_length.append(maxlength8(point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i],point8['x8'][i],point8['y8'][i]))
   min_length.append(minlength8(point8['x1'][i],point8['y1'][i],point8['x2'][i],point8['y2'][i],point8['x3'][i],point8['y3'][i],point8['x4'][i],point8['y4'][i],point8['x5'][i],point8['y5'][i],point8['x6'][i],point8['y6'][i],point8['x7'][i],point8['y7'][i],point8['x8'][i],point8['y8'][i])) 
      
for i in range(len(point9)):
   angle1 = angles(point9['x1'][i],point9['y1'][i],point9['x2'][i],point9['y2'][i],point9['x2'][i],point9['y2'][i],point9['x3'][i],point9['y3'][i])
   angle2 = angles(point9['x2'][i],point9['y2'][i],point9['x3'][i],point9['y3'][i],point9['x3'][i],point9['y3'][i],point9['x4'][i],point9['y4'][i])
   angle3 = angles(point9['x3'][i],point9['y3'][i],point9['x4'][i],point9['y4'][i],point9['x4'][i],point9['y4'][i],point9['x5'][i],point9['y5'][i])
   angle4 = angles(point9['x4'][i],point9['y4'][i],point9['x5'][i],point9['y5'][i],point9['x5'][i],point9['y5'][i],point9['x6'][i],point9['y6'][i])
   angle5 = angles(point9['x5'][i],point9['y5'][i],point9['x6'][i],point9['y6'][i],point9['x6'][i],point9['y6'][i],point9['x7'][i],point9['y7'][i])
   angle6 = angles(point9['x6'][i],point9['y6'][i],point9['x7'][i],point9['y7'][i],point9['x7'][i],point9['y7'][i],point9['x8'][i],point9['y8'][i])
   angle7 = angles(point9['x7'][i],point9['y7'][i],point9['x8'][i],point9['y8'][i],point9['x8'][i],point9['y8'][i],point9['x9'][i],point9['y9'][i])
   angle8 = angles(point9['x8'][i],point9['y8'][i],point9['x9'][i],point9['y9'][i],point9['x9'][i],point9['y9'][i],point9['x1'][i],point9['y1'][i])
   angle9 = angles(point9['x9'][i],point9['y9'][i],point9['x1'][i],point9['y1'][i],point9['x1'][i],point9['y1'][i],point9['x2'][i],point9['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9))
   max_length.append(maxlength9(point9['x1'][i],point9['y1'][i],point9['x2'][i],point9['y2'][i],point9['x3'][i],point9['y3'][i],point9['x4'][i],point9['y4'][i],point9['x5'][i],point9['y5'][i],point9['x6'][i],point9['y6'][i],point9['x7'][i],point9['y7'][i],point9['x8'][i],point9['y8'][i],point9['x9'][i],point9['y9'][i]))
   min_length.append(minlength9(point9['x1'][i],point9['y1'][i],point9['x2'][i],point9['y2'][i],point9['x3'][i],point9['y3'][i],point9['x4'][i],point9['y4'][i],point9['x5'][i],point9['y5'][i],point9['x6'][i],point9['y6'][i],point9['x7'][i],point9['y7'][i],point9['x8'][i],point9['y8'][i],point9['x9'][i],point9['y9'][i])) 
   
for i in range(len(point10)):
   angle1 = angles(point10['x1'][i],point10['y1'][i],point10['x2'][i],point10['y2'][i],point10['x2'][i],point10['y2'][i],point10['x3'][i],point10['y3'][i])
   angle2 = angles(point10['x2'][i],point10['y2'][i],point10['x3'][i],point10['y3'][i],point10['x3'][i],point10['y3'][i],point10['x4'][i],point10['y4'][i])
   angle3 = angles(point10['x3'][i],point10['y3'][i],point10['x4'][i],point10['y4'][i],point10['x4'][i],point10['y4'][i],point10['x5'][i],point10['y5'][i])
   angle4 = angles(point10['x4'][i],point10['y4'][i],point10['x5'][i],point10['y5'][i],point10['x5'][i],point10['y5'][i],point10['x6'][i],point10['y6'][i])
   angle5 = angles(point10['x5'][i],point10['y5'][i],point10['x6'][i],point10['y6'][i],point10['x6'][i],point10['y6'][i],point10['x7'][i],point10['y7'][i])
   angle6 = angles(point10['x6'][i],point10['y6'][i],point10['x7'][i],point10['y7'][i],point10['x7'][i],point10['y7'][i],point10['x8'][i],point10['y8'][i])
   angle7 = angles(point10['x7'][i],point10['y7'][i],point10['x8'][i],point10['y8'][i],point10['x8'][i],point10['y8'][i],point10['x9'][i],point10['y9'][i])
   angle8 = angles(point10['x8'][i],point10['y8'][i],point10['x9'][i],point10['y9'][i],point10['x9'][i],point10['y9'][i],point10['x10'][i],point10['y10'][i])
   angle9 = angles(point10['x9'][i],point10['y9'][i],point10['x10'][i],point10['y10'][i],point10['x10'][i],point10['y10'][i],point10['x1'][i],point10['y1'][i])
   angle10 = angles(point10['x10'][i],point10['y10'][i],point10['x1'][i],point10['y1'][i],point10['x1'][i],point10['y1'][i],point10['x2'][i],point10['y2'][i])
   max_angle.append(max(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10))
   min_angle.append(min(angle1,angle2,angle3,angle4,angle5,angle6,angle7,angle8,angle9,angle10))
   max_length.append(maxlength10(point10['x1'][i],point10['y1'][i],point10['x2'][i],point10['y2'][i],point10['x3'][i],point10['y3'][i],point10['x4'][i],point10['y4'][i],point10['x5'][i],point10['y5'][i],point10['x6'][i],point10['y6'][i],point10['x7'][i],point10['y7'][i],point10['x8'][i],point10['y8'][i],point10['x9'][i],point10['y9'][i],point10['x10'][i],point10['y10'][i]))
   min_length.append(minlength10(point10['x1'][i],point10['y1'][i],point10['x2'][i],point10['y2'][i],point10['x3'][i],point10['y3'][i],point10['x4'][i],point10['y4'][i],point10['x5'][i],point10['y5'][i],point10['x6'][i],point10['y6'][i],point10['x7'][i],point10['y7'][i],point10['x8'][i],point10['y8'][i],point10['x9'][i],point10['y9'][i],point10['x10'][i],point10['y10'][i])) 

mxangle = pd.DataFrame(np.array(max_angle), columns= list('1'))  
mxangle = mxangle.rename(columns={'1':'Max_angle'})    
dataset = pd.concat([dataset, mxangle],axis =1)


mnangle = pd.DataFrame(np.array(min_angle), columns= list('0'))  
mnangle = mnangle.rename(columns={'0':'Min_angle'})    
dataset = pd.concat([dataset, mnangle],axis =1)

mxside = pd.DataFrame(np.array(max_length), columns= list('0'))  
mxside = mxside.rename(columns={'0':'Max_side'})    
dataset = pd.concat([dataset, mxside],axis =1)


mnside = pd.DataFrame(np.array(min_length), columns= list('0'))  
mnside = mnside.rename(columns={'0':'Min_side'})    
dataset = pd.concat([dataset, mnside],axis =1)
    

writer = pd.ExcelWriter('enddata_angside.xlsx')
dataset.to_excel(writer,'Sheet1')
writer.save()      