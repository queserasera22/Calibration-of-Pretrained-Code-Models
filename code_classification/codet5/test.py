import json
import os
import jsonlines

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


dataset_name = "Python800"


def decomposite():
    ## 读取npy文件
    vecs = np.load("../dataset/CodeNet_{}/data_vecs.npy".format(dataset_name))
    print(vecs.shape)

    # pca降维
    pca = PCA(n_components=50, random_state=123456)
    x_pca = pca.fit_transform(vecs)

    print(x_pca.shape)

    ## 保存npy文件
    np.save("../dataset/CodeNet_{}/data_vecs_pca_50.npy".format(dataset_name), x_pca)


def elbow():
    # use elbow method to find the best k

    train_data = np.load("../dataset/CodeNet_{}/data_vecs_pca_50.npy".format(dataset_name))
    SSE = []
    for k in tqdm(range(1, 51)):
        estimator = KMeans(n_clusters=k, random_state=123456)  # 构造聚类器
        estimator.fit(train_data)
        SSE.append(estimator.inertia_)
    X = range(1, 51)
    plt.xlabel('k')
    plt.ylabel('SSE')

    plt.plot(X, SSE, 'o-')
    # plt.show()

    plt.savefig('elbow_{}.png'.format(dataset_name))


def kmeans():
    ## java250分5类, python800分10类

    train_data = np.load("../dataset/CodeNet_{}/data_vecs_pca_50.npy".format(dataset_name))
    kmeans = KMeans(n_clusters=10, random_state=123456)
    cluster = kmeans.fit(train_data)

    # 得到每个样本所属的簇
    y_pred = kmeans.predict(train_data)

    # 打印每个簇的样本数
    for i in range(10):
        print("簇{}的样本数：{}".format(i, np.sum(y_pred == i)))

    # 设置np随机数种子
    np.random.seed(123456)

    # 随机选择2个簇作为测试集， 其余作为训练集
    test_index1 = np.random.randint(0, 10)
    test_index2 = np.random.randint(0, 10)

    print(test_index1, test_index2)

    # 读取jsonl文件, 并切分成训练集和测试集，保存到文件中
    with open("../dataset/CodeNet_{}/data.jsonl".format(dataset_name), "r", encoding="utf-8") as f:
        lines = f.readlines()
        test_lines = []
        train_lines = []
        for index, line in enumerate(lines):
            if y_pred[index] == test_index1 or y_pred[index] == test_index2:
                test_lines.append(json.loads(line.strip()))
            else:
                train_lines.append(json.loads(line.strip()))

        print(len(test_lines))
        print(len(train_lines))

    # 训练集进一步随机划分为训练集和验证集
    train_lines, val_lines = train_test_split(train_lines, test_size=0.2, train_size=0.8, random_state=0, shuffle=True)

    print(len(train_lines))
    print(len(val_lines))

    # 保存到jsonl文件中
    with jsonlines.open("../dataset/CodeNet_{}/OOD/train.jsonl".format(dataset_name), mode='w') as writer:
        writer.write_all(train_lines)

    with jsonlines.open("../dataset/CodeNet_{}/OOD/valid.jsonl".format(dataset_name), mode='w') as writer:
        writer.write_all(val_lines)

    with jsonlines.open("../dataset/CodeNet_{}/OOD/test.jsonl".format(dataset_name), mode='w') as writer:
        writer.write_all(test_lines)


if __name__ == '__main__':
    # decomposite()
    # elbow()
    kmeans()

    # 读取jsonl文件
    # with open("../dataset/CodeNet_Java250/OOD/valid.jsonl", "r", encoding="utf-8") as f:
    #     lines = f.readlines()
    #     for
    #     print(len(lines))
