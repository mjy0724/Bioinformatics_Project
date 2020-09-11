import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np

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
print(len(X))
# divide into Y1 and Y2

tsne = TSNE(n_components=15)
data_3 = tsne.fit_transform(X)
_data = DataFrame(data_3)
_data.to_csv('tSNE_D15.csv', index=False, header=False)

plt.scatter(data_3[:,0], data_3[:,1])
plt.title('tSNE')
plt.show()
