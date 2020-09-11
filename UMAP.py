import numpy as np
import pandas as pd
import umap
import csv
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame



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
                    if c > 0:
                        datarow.append(float(item))
                    c = c + 1
                data.append(datarow)
            r = r + 1

    X = np.array(data).T
    data_fea = X  # 取数据中指标所在的列
    # 标准化
    #data_fea = data_fea.tolist()
    data_mean = data_fea.mean()
    data_std = data_fea.std()
    data_fea = (data_fea - data_mean) / data_std

    # 降维
    umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(data_fea)

    # 归一化
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    umap_data = min_max_scaler.fit_transform(umap_data)

    # 绘制图像
#    plt.figure(figsize=(12, 5))
#    plt.scatter(umap_data[:, 0], umap_data[:, 1])
#    plt.show()

    _data = DataFrame(umap_data)
    _data.to_csv('UMAP.csv', index=False, header=False)

    data_3 = umap_data
    plt.scatter(data_3[:, 0], data_3[:, 1], alpha=0.2)
    plt.title('UMAP')
    plt.show()
