#!/usr/bin/env python
# coding: utf-8

# # Clustering Crypto

# In[1]:


# Initial imports
import pandas as pd
import hvplot.pandas
#from path import Path
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ### Deliverable 1: Preprocessing the Data for PCA

# In[21]:


# Load the crypto_data.csv dataset.
file_path = "crypto_data.csv"
crypto_df = pd.read_csv(file_path, index_col=0)
crypto_df.head()


# In[22]:


# Keep all the cryptocurrencies that are being traded.
new_crypto_df = crypto_df[(crypto_df['IsTrading'] == True)]
new_crypto_df.head()


# In[23]:


#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[24]:


# Keep all the cryptocurrencies that have a working algorithm.
# Drop null rows
new_crypto_df = new_crypto_df.dropna()


# In[25]:


#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[26]:


# Remove the "IsTrading" column. 
new_crypto_df.drop(columns=["IsTrading"], inplace=True)
new_crypto_df.head()


# In[27]:


# Remove rows that have at least 1 null value.
#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[28]:


# Keep the rows where coins are mined.
new_crypto_df = new_crypto_df[(new_crypto_df['TotalCoinsMined'] > 0)]
new_crypto_df.head()


# In[29]:


# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_names_df = pd.DataFrame(new_crypto_df['CoinName'])
crypto_names_df.head()


# In[30]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
new_crypto_df.drop(columns=["CoinName"], inplace=True)
new_crypto_df.head()


# In[31]:


#new_crypto_df.drop(columns=["Unnamed: 0"], inplace=True)
#new_crypto_df.head()


# In[32]:


# Use get_dummies() to create variables for text features.
import pandas as pd
from pathlib import Path

X = pd.get_dummies(new_crypto_df, columns=["Algorithm", "ProofType"])


X.head()


# In[33]:


print(X.dtypes)


# In[34]:


#standardize the dataset using StandardScaler()
#object = StandardScaler()
#object.fit_transform(X) 




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

standardized_data = scaler.fit_transform(X)

print(standardized_data)


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[57]:


# Using PCA to reduce dimension to three principal components.
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit_transform(standardized_data)


# In[110]:


# Create a DataFrame with the three principal components.
pca.columns =['PC1', 'PC2', 'PC3']

data_pca = (pd.DataFrame(pca.fit_transform(standardized_data), index = X.index, columns=['PC1','PC2','PC3'])
           )
data_pca.head(10)


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[111]:


# Create an elbow curve to find the best value for K.

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import hvplot.pandas


# Running K-Means with `k=4`

# In[112]:


import hvplot.pandas

inertia = []
k = list(range(1,11))
for ii in k:
    km = KMeans(n_clusters=ii, random_state=0)
    km.fit(data_pca)
    inertia.append(km.inertia_)

elbow_data = {"k": k, "inertia": inertia}
df_elbow = pd.DataFrame(elbow_data)
    
# Plot
df_elbow.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")


# In[113]:



# Initialize the K-Means model.
model = KMeans(n_clusters=4, random_state=1)

# Fit the model
model.fit(data_pca)

# Predict clusters
predictions = model.predict(data_pca)
predictions


# In[116]:


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
clustered_df = crypto_df.join(data_pca, how='inner')

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
clustered_df['CoinName'] = crypto_names_df['CoinName']

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
clustered_df['Class'] = predictions

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[117]:


# Creating a 3D-Scatter with the PCA data and the clusters
fig = px.scatter_3d(
    clustered_df, 
    x="PC1", 
    y="PC2", 
    z="PC3", 
    color="Class", 
    symbol="Class",
    
    hover_name="CoinName",
    hover_data=["Algorithm"],
    width=800)

fig.update_layout(legend=dict(x=0,y=1))
fig.show()


# In[118]:


# Create a table with tradable cryptocurrencies.
columns = ['CoinName', 'Algorithm', 'ProofType', 'TotalCoinSupply', 'TotalCoinsMined', 'Class']
clustered_df.hvplot.table(columns=columns, sortable=True, selectable=True)


# In[119]:


# Print the total number of tradable cryptocurrencies.
nrows = len(clustered_df.index)
print(f"Total number of tradable cryptocurrencies = {nrows}")


# In[120]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
SupplyMined_scaled = MinMaxScaler().fit_transform(clustered_df[['TotalCoinSupply', 'TotalCoinsMined']])
SupplyMined_scaled


# In[121]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
plot_df = pd.DataFrame(
    data = SupplyMined_scaled,
    index = clustered_df.index,
    columns=["TotalCoinSupply", "TotalCoinsMined"]
)

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
plot_df['CoinName'] = clustered_df['CoinName']

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
plot_df['Class'] = clustered_df['Class']

plot_df.head(10)


# In[122]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
plot_df.hvplot.scatter(x="TotalCoinsMined", y="TotalCoinSupply", hover_cols=["CoinName"], by="Class")


# In[ ]:




