import csv
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np

def name2type(str):
    for i in range(len(feature_name)):
        if feature_name[i] == str:
            return i
    return -1

def doit(_type, _column):
    if _type == 23 or _type == 26 or _type == 27 :
        return
    if _column == -1:
        return
    sumT[_type] = sumT[_type] + 1
    for i in range(len(X)):
        sumX[_type][i] = sumX[_type][i] + X[i][_column]
        #sumT[_type] = sumT[_type] + X[i][_column]

def smooth(N):
    NUM_CLUSTER = 105
    for c in range(NUM_CLUSTER):
        count = np.zeros(65)
        for i in range(N):
            if cluster[i] == c:
                count[int(color[i])] = count[int(color[i])] + 1
        new_c = 0
        for i in range(65):
            if count[i] > count[int(new_c)]:
                new_c = i
        for i in range(N):
            if cluster[i] == c:
                color[i] = new_c

data = []
with open("extra_10000.csv", "r") as inputFile:
    # 读取csv文件,返回的是迭代类型
    reader = csv.reader(inputFile)
    datarow = []
    feature_name = []
    r = 0
    for row in reader:
        if r > 0:  # 去除第1行
            datarow = []
            c = 0
            flag = False
            for item in row:
                if c == 0:
                    feature_name.append(item)
                if c != 0:
                    datarow.append(float(item))
                    if float(item) != 0:
                        flag = True
                c = c + 1
            if c > 1 and flag == True:
                data.append(datarow)
        r = r + 1

X = np.array(data).T

data = []
with open("UMAP.csv", "r") as inputFile:
    # 读取csv文件,返回的是迭代类型
    reader = csv.reader(inputFile)
    datarow = []
    r = 1
    for row in reader:
        if r > 0:  # 去除第1行
            datarow = []
            c = 0
            for item in row:
                datarow.append(float(item))
                c = c + 1
            data.append(datarow)
        r = r + 1

Y = np.array(data)
N=len(Y)
data_3 = np.zeros((N,2))
for i in range(len(Y)):
    data_3[i][0] = Y[i][0]
    data_3[i][1] = Y[i][1]
'''
data = []
with open("feature_name.csv", "r") as inputFile:
    # 读取csv文件,返回的是迭代类型
    reader = csv.reader(inputFile)
    datarow = []
    r = 1
    for row in reader:
        if r > 0:  # 去除第1行
            datarow = []
            c = 0
            for item in row:
                if c == 0:
                    datarow.append(str(item))
                c = c + 1
            data.append(datarow)
        r = r + 1
feature_name = np.array(data)
'''
# divide into Y1 and Y2
sumX = np.zeros((65, len(X)))

data = []
sumT = np.zeros(65)
with open("CellMarker1.csv", "r") as inputFile:
    reader = csv.reader(inputFile)
    datarow = []
    r = 0
    for row in reader:
        if r > 0:  # 去除第1行
            datarow = []
            c = 0
            for item in row:
                datarow.append(item)
                c = c + 1
            data.append(datarow)
            _type = int(datarow[0])
            _column = name2type(str(datarow[8]))
            doit(_type, _column)
        r = r + 1

eps=1e-2
N=len(X)
color=np.zeros(N)
volume=np.zeros(N)
for i in range(len(X)):
    ctype=0
    for type in range(61):
        if (sumT[type] > eps):
            ctype = type
    for type in range(61):
        if sumT[type] > eps and sumX[type][i] / sumT[type] > sumX[ctype][i] / sumT[ctype]:
            ctype = type
    color[i] = ctype
    volume[i] = sumX[ctype][i]

colors=set()
N = len(data_3)
for i in range(N):
    colors.add(color[i])
print(colors)

'''
cnt=0
for i in range(N):
    Sum = 0
    for j in range(len(X[i])):
        Sum = Sum + X[i][j]
    if Sum < 0.01:
        cnt = cnt + 1
print('# zero cells:', cnt)'''
maxV = np.max(volume)/2



data = []
cluster = np.zeros(N)
with open("idxkmean(3).csv", "r") as inputFile:
    reader = csv.reader(inputFile)
    datarow = []
    r = 1
    for row in reader:
        if r > 0:  # 去除第1行
            datarow = []
            c = 0
            for item in row:
                datarow.append(item)
                c = c + 1
            data.append(datarow)
        r = r + 1

for i in range(N):
    cluster[i] = int(data[i][0])
    print(i, cluster[i])

smooth(N)



s=[]
for i in range(N):
    s.append(5)
#plt.scatter(data_3[:,0],data_3[:,1], color='#000000', s=s,alpha=0.2)
pointsize = 50
#type2
poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 2 or color[i] == 1 or color[i] == 0:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#DC143C',s=pointsize,alpha=0.2,label='Acinarcell')

#type5
poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 5 or color[i] == 6 or color[i] == 7:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#FF8C00',s=pointsize,alpha=0.2,label='Alphacell')

#type20
poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 11 or color[i] == 12 or color[i] == 14 or color[i] == 20 or color[i] == 21 or color[i] == 19:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#FFD700',s=pointsize,alpha=0.2,label='Betacell')

'''
poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 23 or color[i] == 26 or color[i] == 27:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#00FF00',s=pointsize,alpha=0.2,label='Cancerstemcell')
'''

poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 31 or color[i] == 30:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#00FFFF',s=pointsize,alpha=0.2, label='Deltacell')


poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 33:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#1E90FF',s=pointsize,alpha=0.2,label='Ductalcell')



poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 43:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#000080',s=pointsize,alpha=0.2,label='Mesenchymalcell')



poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 34 or color[i] == 36 or color[i] == 37:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#9400D3',s=pointsize,alpha=0.2,label='Endocrineprogenitorcell')


poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 42:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(10.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#A9A9A9',s=pointsize,alpha=0.2,label='Fibroblast')

poi_x = []
poi_y = []
s = []
for i in range(N):
    if color[i] == 60 or color[i] == 48:
        poi_x.append(data_3[i][0])
        poi_y.append(data_3[i][1])
        s.append(max(20.0, volume[i]/maxV * 10))
plt.scatter(poi_x, poi_y,color='#00FF00',s=pointsize,alpha=0.2,label='Pancreaticpolypeptidecell')

plt.legend()
plt.show()

