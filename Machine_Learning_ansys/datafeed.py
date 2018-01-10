# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:19:25 2017

@author: kgavini

Store values into a data frame and export them back to excel
"""


import pandas as pd
import numpy as np
df = pd.read_csv('10point.csv')
xval = df.loc[:,'x(j)']
df = pd.DataFrame(np.array(xval).reshape(145,11), columns = list("1234567890a"))
df = df.rename(columns={'1':'x1','2':'x2','3':'x3','4':'x4','5':'x5','6':'x6','7':'x7','8':'x8','9':'x9','0':'x10','a':'x11'})
df = df.drop('x11',1)

df1 = pd.read_csv('10point.csv')
yval = df1.loc[:,'y(j)']
df1 = pd.DataFrame(np.array(yval).reshape(145,11), columns = list("1234567890a"))
df1 = df1.rename(columns={'1':'y1','2':'y2','3':'y3','4':'y4','5':'y5','6':'y6','7':'y7','8':'y8','9':'y9','0':'y10','a':'y11'})
df1 = df1.drop('y11',1)
data = pd.concat([df, df1], axis=1)
writer = pd.ExcelWriter('10point.xlsx')
data.to_excel(writer,'Sheet1')
writer.save()



"""
Append mm to Q3D script
"""
x = []
for i in xval:
    x.append(str(i)+'mm')
y=[]
for i in yval:
    y.append(str(i)+'mm')    
    

"""
Store Capacitance values from exported excel sheets
"""    
capacitance = []    
for i in range(2,1587,11):
    x = open('result10'+str(i)+'.csv')
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
    
    
capdata = pd.DataFrame(np.array(capacitance).reshape(145,1),columns =list(['c']))
capdata = capdata.rename(columns={'c':'capacitance'})
data = pd.concat([data, capdata], axis=1) 


"""
Find area of a polygon
"""

def poly_area1(xy):
    
    xy1 = np.roll(xy,-1,axis = 0) # shift by -1
    return 0.5*abs(np.inner(xy1[:,0] - xy[:,0],xy1[:,1] + xy[:,1]))

#xy = np.array([[56.19,46.35],[52.14,42],[55.08,44.97]])
#poly_area1(xy)

area = []
for i in range(145):
    xy = np.array([[data['x1'][i],data['y1'][i]],[data['x2'][i],data['y2'][i]],[data['x3'][i],data['y3'][i]],[data['x4'][i],data['y4'][i]],[data['x5'][i],data['y5'][i]],[data['x6'][i],data['y6'][i]],[data['x7'][i],data['y7'][i]],[data['x8'][i],data['y8'][i]],[data['x9'][i],data['y9'][i]],[data['x10'][i],data['y10'][i]]])
    area.append(poly_area1(xy))
    
areadata = pd.DataFrame(np.array(area).reshape(145,1), columns = list("A")) 
areadata = areadata.rename(columns={'A':'area'}) 
data = pd.concat([data, areadata], axis=1)  




"""
Removing intersecting polygons
"""

poly = pd.read_csv('10point1.csv')

poly.count

poly = poly.dropna(axis=0, how='all')

for i in [200,198,195,190,189,185,184,182,181,179,178,171,170,167,160,158,151,149,145,143,140,139,138,135,132,126,124,120,118,117,114,110,107,106,101,98,95,87,85,75,68,66,60,58,57,56,51,47,42,41,39,38,18,11,9]:
 poly = poly[poly['count'] != i]    

writer = pd.ExcelWriter('10point.xlsx')
poly.to_excel(writer,'Sheet1')
writer.save()
    
a = 5.943483826847684
b = 4.179054917083524
c = 1.7710166571774546
s = (a + b + c) / 2  

area = (s*(s-a)*(s-b)*(s-c)) ** 0.5

