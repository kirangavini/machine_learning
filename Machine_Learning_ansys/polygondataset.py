# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:45:19 2017

@author: kgavini
"""

import pandas as pd
import numpy as np

datathree = pd.read_csv('3point.csv')

threearray = []
for i in range(500):
    threearray.append(3)
    
threedata = pd.DataFrame(np.array(threearray), columns= list('3'))    
data = pd.concat([datathree, threedata],axis =1)
    
data = data.drop(['Unnamed: 0','x1','x2','x3','y1','y2','y3'], axis=1)    
data = data.rename(columns={'3':'No of points'})  
data = data[['No of points','area','capacitance']]  
    

datafour = pd.read_csv('4point.csv')

fourarray = []
for i in range(500):
    fourarray.append(4)
    
fourdata = pd.DataFrame(np.array(fourarray), columns= list('4'))    
data1 = pd.concat([datafour, fourdata],axis =1)
    
data1 = data1.drop(['Unnamed: 0','x1','x2','x3','x4','y1','y2','y3','y4'], axis=1)    
data1 = data1.rename(columns={'4':'No of points'})  
data1 = data1[['No of points','area','capacitance']] 


datafive = pd.read_csv('5point.csv')

fivearray = []
for i in range(190):
    fivearray.append(5)
    
fivedata = pd.DataFrame(np.array(fivearray), columns= list('5'))    
data2 = pd.concat([datafive, fivedata],axis =1)
    
data2 = data2.drop(['Unnamed: 0','x1','x2','x3','x4','x5','y1','y2','y3','y4','y5'], axis=1)    
data2 = data2.rename(columns={'5':'No of points'})  
data2 = data2[['No of points','area','capacitance']] 


datasix = pd.read_csv('6point.csv')

sixarray = []
for i in range(186):
    sixarray.append(6)
    
sixdata = pd.DataFrame(np.array(sixarray), columns= list('6'))    
data3 = pd.concat([datasix, sixdata],axis =1)
    
data3 = data3.drop(['Unnamed: 0','x1','x2','x3','x4','x5','x6','y1','y2','y3','y4','y5','y6'], axis=1)    
data3 = data3.rename(columns={'6':'No of points'})  
data3 = data3[['No of points','area','capacitance']] 


dataseven = pd.read_csv('7point.csv')

sevenarray = []
for i in range(184):
    sevenarray.append(7)
    
sevendata = pd.DataFrame(np.array(sevenarray), columns= list('7'))    
data4 = pd.concat([dataseven, sevendata],axis =1)
    
data4 = data4.drop(['Unnamed: 0','x1','x2','x3','x4','x5','x6','x7', 'y1','y2','y3','y4','y5','y6','y7'], axis=1)    
data4 = data4.rename(columns={'7':'No of points'})  
data4 = data4[['No of points','area','capacitance']] 


dataeight = pd.read_csv('8point.csv')

eightarray = []
for i in range(174):
    eightarray.append(8)
    
eightdata = pd.DataFrame(np.array(eightarray), columns= list('8'))    
data5 = pd.concat([dataeight, eightdata],axis =1)
    
data5 = data5.drop(['Unnamed: 0','x1','x2','x3','x4','x5','x6','x7','x8', 'y1','y2','y3','y4','y5','y6','y7','y8'], axis=1)    
data5 = data5.rename(columns={'8':'No of points'})  
data5 = data5[['No of points','area','capacitance']] 


datanine = pd.read_csv('9point.csv')

ninearray = []
for i in range(164):
    ninearray.append(9)
    
ninedata = pd.DataFrame(np.array(ninearray), columns= list('9'))    
data6 = pd.concat([datanine, ninedata],axis =1)
    
data6 = data6.drop(['Unnamed: 0','x1','x2','x3','x4','x5','x6','x7','x8','x9', 'y1','y2','y3','y4','y5','y6','y7','y8','y9'], axis=1)    
data6 = data6.rename(columns={'9':'No of points'})  
data6 = data6[['No of points','area','capacitance']] 

dataten = pd.read_csv('10point.csv')

tenarray = []
for i in range(145):
    tenarray.append(10)
    
tendata = pd.DataFrame(np.array(tenarray), columns= list('0'))    
data7 = pd.concat([dataten, tendata],axis =1)
    
data7 = data7.drop(['Unnamed: 0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10', 'y1','y2','y3','y4','y5','y6','y7','y8','y9','y10'], axis=1)    
data7 = data7.rename(columns={'0':'No of points'})  
data7 = data7[['No of points','area','capacitance']] 

enddata = pd.concat([data, data1,data2,data3,data4,data5,data6,data7],axis =0,ignore_index = True)

writer = pd.ExcelWriter('enddata.xlsx')
enddata.to_excel(writer,'Sheet1')
writer.save()