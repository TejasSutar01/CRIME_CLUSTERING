sum(# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:58:04 2020

@author: tejas
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
df=pd.read_csv("D:\\TEJAS FORMAT\\EXCELR ASSIGMENTS\\COMPLETED\\CLUSTERING\\CRIME DATA\\crime_data.csv")
 
############Normalizing the function############
def norm_func(i):
    x=(i-i.mean()/i.std())
    return x

df_norm=norm_func(df.iloc[:,1:])

######################creating the dendogram##################
z=linkage(df_norm,method="complete", metric="euclidean")
sch.dendrogram(z,leaf_rotation=0.,leaf_font_size=8.,)

#########Defining the clusters############
from sklearn.cluster import AgglomerativeClustering
model=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="complete").fit(df_norm)
clusters_labels=pd.Series(model.labels_)
df["Clusters"]=clusters_labels
df=df.iloc[:,[5,0,1,2,3,4]]
df.iloc[:,2:].groupby(df.Clusters).median()


###########importing to csv#############
df.to_csv("crime.csv",encoding="utf-8")
