# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:08:27 2017

@author: kgavini
"""

 
import os
import shutil
import pandas as pd
import math
src_dir = "F:\iso"
dst_dir = "F:\python\polygon_prediction\cap"
for root, dirs, files in os.walk(src_dir):
    for f in files:
        if f.endswith('.csv'):
            shutil.copy(os.path.join(root,f), dst_dir)   


capacitance = []    
for i in range(2,40000,4):
    x = open('result'+str(i)+'.csv')
    x = x.read() 
    x = x.split()
    capvalue = x[16][10:]
    capvalue1 =x[17]
    try:
        cval = float(capvalue)
    except ValueError:
        capvalue = capvalue
           
    try:
        cval = float(capvalue1)
    except ValueError:
        capvalue1 = capvalue1 
    capacitance.append(cval)
dataset = pd.DataFrame({'capacitance':capacitance})   


import numpy as np
df = pd.read_csv('xaxis.csv')
xval = df.loc[:,'x(j)']
df = pd.DataFrame(np.array(xval).reshape(10000,4), columns = list("1234"))
df = df.rename(columns={'1':'x1','2':'x2','3':'x3','4':'x4'})
df = df.drop('x4',1)

df1 = pd.read_csv('yaxis.csv')
yval = df1.loc[:,'y(j)']
df1 = pd.DataFrame(np.array(yval).reshape(10000,4), columns = list("1234"))
df1 = df1.rename(columns={'1':'y1','2':'y2','3':'y3','4':'y4'})
df1 = df1.drop('y4',1)
data = pd.concat([df, df1, dataset], axis=1)


"""
Find area of a polygon
"""

def poly_area1(xy):
    
    xy1 = np.roll(xy,-1,axis = 0) # shift by -1
    return 0.5*abs(np.inner(xy1[:,0] - xy[:,0],xy1[:,1] + xy[:,1]))

#xy = np.array([[56.19,46.35],[52.14,42],[55.08,44.97]])
#poly_area1(xy)

area = []
for i in range(10000):
    xy = np.array([[data['x1'][i],data['y1'][i]],[data['x2'][i],data['y2'][i]],[data['x3'][i],data['y3'][i]]])
    area.append(poly_area1(xy))
    
areadata = pd.DataFrame(np.array(area).reshape(10000,1), columns = list("A")) 
areadata = areadata.rename(columns={'A':'area'}) 
data = pd.concat([data, areadata], axis=1)  


"""
Angles and side extrimities

"""
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

def perimeter(points):
    """ returns the length of the perimiter of some shape defined by a list of points """
    distances = get_distances(points)

    length = 0
    for distance in distances:
        length = length + distance

    return length


def get_distances(points):
    """ convert a list of points into a list of distances """
    i = 0
    distances = []
    for i in range(len(points)-1):
        point = points[i]
        next_point = points[i+1]
        x0 = point[0]
        y0 = point[1]
        x1 = next_point[1]
        y1 = next_point[1]

        point_distance = get_distance(x0, y0, x1, y1)
        distances.append(point_distance)
    return distances    


def get_distance(x0, y0, x1, y1):
    """ use pythagorean theorm to find distance between 2 points """
    a = x0 - x1
    b = y0 - y1
    c_2 = a*a + b*b

    return c_2 ** (1/2)

max_angle= []
min_angle= []
max_length = []
min_length = []
perimeters = []

for i in range(len(data)):
    
  angle1 = angles(data['x1'][i],data['y1'][i],data['x2'][i],data['y2'][i],data['x2'][i],data['y2'][i],data['x3'][i],data['y3'][i])
  angle2 = angles(data['x2'][i],data['y2'][i],data['x3'][i],data['y3'][i],data['x3'][i],data['y3'][i],data['x1'][i],data['y1'][i])
  angle3 = angles(data['x3'][i],data['y3'][i],data['x1'][i],data['y1'][i],data['x1'][i],data['y1'][i],data['x2'][i],data['y2'][i])
  max_angle.append(max(angle1,angle2,angle3))
  min_angle.append(min(angle1,angle2,angle3))
  max_length.append(maxside(data['x1'][i],data['y1'][i],data['x2'][i],data['y2'][i],data['x3'][i],data['y3'][i]))
  min_length.append(minside(data['x1'][i],data['y1'][i],data['x2'][i],data['y2'][i],data['x3'][i],data['y3'][i]))
  perimeters.append(perimeter([[data['x1'][i],data['y1'][i]],[data['x2'][i],data['y2'][i]],[data['x3'][i],data['y3'][i]]]))
dataset = pd.DataFrame({'max_angle':max_angle,'min_angle':min_angle,'max_length':max_length,'min_length':min_length})   

data = pd.concat([data, dataset], axis=1) 



    
dataset = pd.DataFrame({'capacitance':capacitance})    
    
    
data.to_csv('isosclesdata.csv', sep=',')