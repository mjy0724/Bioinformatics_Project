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

data_3 = np.array(data)
N=len(data_3)

data_new = []
for i in range(N):
    if i != 1728:
        data_new.append(data_3[i])
data_new = np.array(data_new)
#_data = DataFrame(data_new)
#_data.to_csv('tSNE_xx.csv', index=False, header=False)

plt.scatter(data_new[:,0], data_new[:,1], alpha=0.2)
plt.title('tSNE')
plt.show()





