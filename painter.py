import csv
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from sklearn.manifold import TSNE
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np

def upd(c,sum):
    return max(sum, sumX[c][i]/max(sumT[c],eps))

def name2type(str):
    for i in range(len(feature_name)):
        if feature_name[i] == str:
            return i
    return -1

def doit(_type, _column):
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
with open("tSNE_1.csv", "r") as inputFile:
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

plt.scatter(data_3[:][0],data_3[:][1],color='#C0C0C0',s=50,alpha=0.1)

for i in range(N):
    sum = 0
    sum = upd(48,sum)
    sum = upd(60,sum)
    plt.scatter(data_3[i][0], data_3[i][1], color='#A52A2A', s=50, alpha = 0.5*sum/sumX[int(color[i])][i]*sumT[int(color[i])])
plt.title('Pancreaticpolypeptidecell')
#plt.scatter(data_3[:,0],data_3[:,1], color='#000000', s=s,alpha=0.2)
plt.legend()
plt.show()


