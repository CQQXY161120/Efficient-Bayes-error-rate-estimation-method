# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:37:46 2023

@author: Anna
"""

import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist,squareform
from sklearn.semi_supervised import LabelSpreading
from memory_profiler import profile
import faiss

class FaissKNeighbors:
    def __init__(self, k=20):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
    
def divided_data(data,label):
    s = 1
    cluster_list = []     #homogeneous cluster list
    cluster_centers = []  #homogeneous cluster centers list
    cluster_temp = [np.arange(len(label))]
    k=0
    while cluster_temp:
        k += 1
        ind_cur = cluster_temp[0]
        if len(ind_cur)>s:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data[ind_cur,:])
            cluster1_ind = ind_cur[np.where(kmeans.labels_==0)[0]]
            cluster2_ind = ind_cur[np.where(kmeans.labels_==1)[0]]
            cluster1, cluster2 = kmeans.cluster_centers_
        
            if len(set(label[cluster1_ind].ravel()))==1:
                cluster_list.append(cluster1_ind)
                cluster_centers.append(cluster1)
            else:
                if len(cluster1_ind)>s:
                    cluster_temp.append(cluster1_ind)
            if len(set(label[cluster2_ind].ravel()))==1:
                cluster_list.append(cluster2_ind)
                cluster_centers.append(cluster2)
            else:
                if len(cluster2_ind)>s:
                    cluster_temp.append(cluster2_ind)
            del cluster_temp[0]
        else:
            del cluster_temp[0]
    return cluster_list


def EF_BER_F(data,label):
    cluster_list = []     #homogeneous cluster list
    cluster_temp = [np.arange(len(label))]
    n = len(label)       #dataset size
    cluster_temp = [np.arange(n)]
    s=4
    while cluster_temp:
        ind_cur = cluster_temp[0]
        if len(ind_cur)>s:
            kmeans = faiss.Kmeans(d=data[ind_cur,:].shape[1], k=2, niter=300)
            kmeans.train(data[ind_cur,:].astype(np.float32))
            
            cluster1_ind = ind_cur[np.where(kmeans.index.search(data[ind_cur,:].astype(np.float32), 1)[1]==0)[0]]
            cluster2_ind = ind_cur[np.where(kmeans.index.search(data[ind_cur,:].astype(np.float32), 1)[1]==1)[0]]
            cluster1, cluster2 = kmeans.centroids
        
            if len(set(label[cluster1_ind].ravel()))==1:
                cluster_list.append(cluster1_ind)       
            else:
                if len(cluster1_ind)>s:
                    if set(cluster1_ind)&set(ind_cur)!=set(cluster1_ind)|set(ind_cur):
                        cluster_temp.append(cluster1_ind)
            if len(set(label[cluster2_ind].ravel()))==1:
                cluster_list.append(cluster2_ind)
            else:
                if len(cluster2_ind)>s:
                    if set(cluster2_ind)&set(ind_cur)!=set(cluster2_ind)|set(ind_cur):
                        cluster_temp.append(cluster2_ind)
            del cluster_temp[0]
        else:
            del cluster_temp[0]
    

    conf_index = []
    for i in range(len(cluster_list)):
        list_i = cluster_list[i]
        if len(list_i)>s:
            conf_index.extend(list_i)
    non_conf_index = list(set(list(np.arange(len(label))))-set(list(conf_index)))
    if non_conf_index:
        neigh = FaissKNeighbors()
        neigh.fit(data[conf_index,:], label[conf_index])
        label_pre = neigh.predict(data[non_conf_index,:])
        ber = len(np.where(label[non_conf_index]-label_pre!=0)[0])/len(label)
    else:
        ber=0
    return ber

def EF_BER_S(data,label):
    s = 4
    cluster_list = []     #homogeneous cluster list
    cluster_centers = []  #homogeneous cluster centers list
    cluster_temp = [np.arange(len(label))]
    k=0
    while cluster_temp:
        k += 1
        ind_cur = cluster_temp[0]
        if len(ind_cur)>s:
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(data[ind_cur,:])
            cluster1_ind = ind_cur[np.where(kmeans.labels_==0)[0]]
            cluster2_ind = ind_cur[np.where(kmeans.labels_==1)[0]]
            cluster1, cluster2 = kmeans.cluster_centers_
        
            if len(set(label[cluster1_ind].ravel()))==1:
                cluster_list.append(cluster1_ind)
                cluster_centers.append(cluster1)
            else:
                if len(cluster1_ind)>s:
                    cluster_temp.append(cluster1_ind)
            if len(set(label[cluster2_ind].ravel()))==1:
                cluster_list.append(cluster2_ind)
                cluster_centers.append(cluster2)
            else:
                if len(cluster2_ind)>s:
                    cluster_temp.append(cluster2_ind)
            del cluster_temp[0]
        else:
            del cluster_temp[0]
    
    conf_index = []
    for i in range(len(cluster_list)):
        list_i = cluster_list[i]
        if len(list_i)>s:
            conf_index.extend(list_i)
    non_conf_index = list(set(list(np.arange(len(label))))-set(list(conf_index)))
    if non_conf_index:
        neigh = KNeighborsClassifier()
        neigh.fit(data[conf_index,:], label[conf_index])
        label_pre = neigh.predict(data[non_conf_index,:])
        ber = len(np.where(label[non_conf_index]-label_pre!=0)[0])/len(label)
    else:
        ber=0
    return ber

def BN_BER(Data, Label):
    """
    Compute BER based on graph search

    Parameters
    ----------
    Data : narray
        input matrix.
    Label : narray
        label.
    k : int
        control the ratio that recognized as noisy samples in dataset.

    Returns
    -------
    TYPE
        connected regions, noise index, G-BER.

    """
    
    visited, result, conf = set(), [], []
    n = len(Label)
    Store_Matrix = np.zeros((n,n))
    ### Index matrix that store the neighbor indexs of samples
    
    Sorted_Index = np.argsort(squareform(pdist(Data,'euclidean'),force='no',checks=True))
    for i in range(n):
        label_i = Label[i]
        for j in range(n):
            if label_i == Label[Sorted_Index[i][j]]:
                Store_Matrix[i][Sorted_Index[i][j]] = 1
            else:
                break
    Store_Matrix = Store_Matrix.astype(int)
    ### Insert reverse neighbor information
    for i in range (n):
        Store_Matrix[i,np.nonzero(Store_Matrix[:,i])[0]] = 1
    ###BFS
    def BFS(Store_Matrix,s):
        queue = []
        queue.append(s) # 向list添加元素，用append()
        seen = set() # 此处为set, python里set用的是hash table, 搜索时比数组要快。
        seen.add(s) # 向set添加函数，用add()
        while (len(queue) > 0):
            vertex = queue.pop(0)  #提取队头
            nodes = np.nonzero(Store_Matrix[vertex,:])  #获得队头元素的邻接元素
            for w in list(nodes)[0]:
                if w not in seen:
                    queue.append(w) #将没有遍历过的子节点入队
                    seen.add(w) #标记好已遍历 
        return seen    
        
    for i in range(n):
        if  i not in visited:
            seen = BFS(Store_Matrix,i)
            visited = visited | set(seen)
            result.append(seen)  
    result_s = sorted(result, key = lambda inx : -len(inx) )
    
    for i in range(len(result_s)):
        if len(result_s[i]) > 5:
            conf.extend(result_s[i])
    unconf = list(set(list(range(n)))-set(conf))
    if len(unconf)!=0: 
        label_prop_model = LabelSpreading()
        label_c = np.copy(Label)
        label_c[unconf] = -1
        label_prop_model.fit(Data, label_c)
        label_p = label_prop_model.predict(Data[unconf,:])
        ber = len(np.where(Label[unconf]-label_p!=0)[0])/n
    else:
        ber=0
    
    return ber

def Load_Data(filename):
    Datasets = np.loadtxt(filename,delimiter=',')
    [InstanceNum,AttributeNum] = Datasets.shape
    DataMatrix = Datasets[:,:AttributeNum-1]
    DataLabel = (Datasets[:,AttributeNum-1]).astype(int)
    return DataMatrix,DataLabel


def FRT (Data, label, n):
    Dis_Mat = squareform(pdist(Data,'euclidean'),force='no',checks=True)
    res = []
    seleted_node = [0]
    candidate_node = [i for i in range(1, n)]
    count = 0
    while len(candidate_node) > 0:
        begin, end, minweight = 0, 0, 999999999
        for i in seleted_node:
            for j in candidate_node:
                if Dis_Mat[i][j] < minweight:
                    minweight = Dis_Mat[i][j]
                    begin = i
                    end = j
        res.append([begin, end, minweight])
        seleted_node.append(end)
        candidate_node.remove(end)
    for i in range(len(res)):
        node1,node2 = res[i][0],res[i][1]
        if label[node1]-label[node2]:
            count += 1
    return count