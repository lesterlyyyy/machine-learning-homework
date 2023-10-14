import random
from data_gen import *
import numpy as np
test_data=np.array([[180,72],
           [150,50],
           [190,80],
            [200,90],
            [140,30]])

import matplotlib.pyplot as plt


train_data=read_data()
train_data,data_min,data_max =normalize(train_data)
"""
聚类的类别数
"""
C=2
def C_means(x,C,max_iter=10,epison=1e-9):
    """
    :param X:n*dim
    :param C: 类别个数
    :return:
    """
    dims=np.arange(0,len(x)).tolist()
    dim=x.shape[1]
    r= random.sample(dims,C)
    intial_centers=x[r]
    r =[np.sum((x-center)**2,axis=1) for center in intial_centers]
    t=np.vstack([i for i in r]).T
    labels=np.argmin(t,axis=1)
    centers=np.zeros((C,dim))
    dic={c:[] for c in range(C)}
    for i in range(len(x)):
        dic[labels[i]].append(x[i])
    print(dic)
    for i in range(len(centers)):
        for item in dic[i]:
            centers[i]+=item
        centers[i]/=len(dic[i])
    print('*'*50,centers)
    je= Je(x,labels,centers,C)
    print(je)
    new_centers=np.zeros_like(centers)
    for iter in range(max_iter):
        for i in range(len(x)):
            mask= labels == labels[i]
            N_k=len(labels[mask])
            m_k= centers[labels[i]]
            if N_k==1:
                continue
            for c in range(C):
                if c==labels[i]:
                    continue
                N_j=len(labels[labels==c])
                m_j=centers[c]

                p_j=N_j/(N_j+1)*np.linalg.norm(x[i]-m_j)
                p_k=N_k/(N_k-1)*np.linalg.norm(x[i]-m_k)
                if p_j<p_k:
                    if np.linalg.norm(new_centers-centers)<epison and je<1000:
                        return labels,centers
                    centers[labels[i]]=m_k+1/(N_k-1)*(m_k-x[i])
                    centers[c]=centers[c]+1/(N_j+1)*(x[i]-m_j)
                    labels[i] = c
                    je = je+p_j-p_k
                    print('*' * 50, centers)
                    print(je)
                    new_centers=centers
                    break
    return labels, centers

def Je(x,labels,centers,C):
    result=0
    for c in range(C):
        result+=np.linalg.norm(x[labels==c]-centers[c])
    return result

# labels,centers=  C_means(train_data,C,max_iter=10)
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt
from utils import draw_disturbution


je=[]
def draw_Je(train_data,C):
    for c in range(1,C):
        kmeans=KMeans(n_clusters=c)
        kmeans.fit(train_data)
        je.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1,10),je,label='Je')
    plt.ylabel('Je')
    plt.xlabel('C')
    plt.legend()
    plt.show()
# 创建一个C均值聚类模型
# init_data=[[165,50,8.5,3000],[175,70,7.5,4000]]
init_data=[[150,50,8.5,3000],[150,50,7.5,3000]]
kmeans = KMeans(n_clusters=2, init='random')
# 定义你的数据集

# 将数据集传递给聚类模型进行训练

kmeans.fit(train_data)
print(unnormalize( kmeans.cluster_centers_,data_min,data_max))
print(kmeans.n_iter_)
draw_disturbution(unnormalize(train_data,min=data_min,max=data_max),kmeans.labels_)
