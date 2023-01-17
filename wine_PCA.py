# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 23:56:30 2022

@author: Mohd Ariz Khan
"""
# Import the data
import pandas as pd
df = pd.read_csv("wine.csv")
df

# Get information of the dataset
df.info()
df.isnull().any()
print('The shape of our data is:', df.shape)
print(df.describe())
df.head()

# Value counts 
df['Type'].value_counts()

# Drop out the first column
df = df.iloc[:,1:]
df.shape
df.info()

# Converting data to numpy array
df_array = df.values
df_array

# Normalizing the  numerical data
from sklearn.preprocessing import scale
df_norm = scale(df_array)
df_norm

# Applying PCA Fit Transform to dataset
from sklearn.decomposition import PCA
pca = PCA()
pca_values  = pca.fit_transform(X_scale)
pca_values

# PCA Components matrix or convariance Matrix
pca.components_


# The amount of variance that each PCA has
var = pca.explained_variance_ratio_
var

# Cummulative variance of each PCA
import numpy as np
Var = np.cumsum(np.round(var,decimals= 4)*100)
Var

plt.plot(Var,color="orange")

# Final Dataframe
final_df = pd.concat([df['Type'],pd.DataFrame(pca_values[:,0:3], columns=['PC1','PC2','PC3'])],axis=1)
final_df

# Visualization of PCAs
import seaborn as sns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,12))         
sns.scatterplot(data = final_df)

sns.scatterplot(data = final_df, x='PC1', y='PC2', hue='Type')
    
pca_values[: ,0:1]

x = pca_values[:,0:1]
y = pca_values[:,1:2]
plt.scatter(x, y)

#=============================================================================
#                Checking with other Clustering Algorithms
#=============================================================================

# Hierarchical Clustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize =(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(df_norm, method='complete')) 

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')
Y = cluster.fit_predict(df_norm)

Y = pd.DataFrame(Y)
Y[0].value_counts()

# Adding clusters to dataset
df_new = df.copy()
df_new['clustersid'] = cluster.labels_
df_new

plt.figure(figsize=(10, 7))  
plt.scatter(df_norm[:,0], df_norm[:,1], c=cluster.labels_, cmap='rainbow')  

#================================================================================
# K-Means Clustering
from sklearn.cluster import KMeans

# As we already have normalized data
# Use Elbow Graph to find optimum number of  clusters (K value) from K values range
# The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion WCSS 
# random state can be anything from 0 to 42, but the same number to be used everytime,so that the results don't change.

# within-cluster sum-of-squares criterion 
wcss=[]
for i in range (1,6):
    kmeans = KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)

# Plot K values range vs WCSS to get Elbow graph for choosing K (no. of clusters)
plt.plot(range(1,6),wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')



