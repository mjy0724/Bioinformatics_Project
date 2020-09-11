import csv
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import pandas as pd



if __name__ == '__main__':
    # 读取csv文件
    data = []
    with open("extra_10000.csv", "r") as inputFile:
        # 读取csv文件,返回的是迭代类型
        reader = csv.reader(inputFile)
        datarow = []
        r = 0
        for row in reader:
            if r > 0:  # 去除第1行
                datarow = []
                c = 0
                for item in row:
                    if c>0:
                        datarow.append(float(item))
                    c = c + 1
                data.append(datarow)
            r = r + 1

    X = np.array(data).T


    fa = FactorAnalysis(n_components = 2)  # sklearn包中的fa算法
    fa.fit(X)
    data_3 = fa.transform(X)
    #for i in range(len(data_3)):
    #    print(i, data_3[i][0], data_3[i][1],data_3[i][2])

    #_data = DataFrame(data_3)
    #_data.to_csv('FA.csv', index=False, header=False)
    plt.scatter(data_3[:, 0], data_3[:, 1], alpha=0.2)
    plt.title('FA')
    plt.show()


