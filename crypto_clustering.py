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

# In[2]:


# Load the crypto_data.csv dataset.
file_path = "crypto_data.csv"
crypto_df = pd.read_csv(file_path)
crypto_df.head()


# In[3]:


# Keep all the cryptocurrencies that are being traded.
new_crypto_df = crypto_df[(crypto_df['IsTrading'] == True)]
new_crypto_df.head()


# In[4]:


#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[5]:


# Keep all the cryptocurrencies that have a working algorithm.
# Drop null rows
new_crypto_df = new_crypto_df.dropna()


# In[6]:


#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[7]:


# Remove the "IsTrading" column. 
new_crypto_df.drop(columns=["IsTrading"], inplace=True)
new_crypto_df.head()


# In[8]:


# Remove rows that have at least 1 null value.
#find null values
for column in new_crypto_df.columns:
    print(f"Column{column}) has {new_crypto_df[column].isnull().sum()} null values")


# In[9]:


# Keep the rows where coins are mined.
new_crypto_df = new_crypto_df[(new_crypto_df['TotalCoinsMined'] > 0)]
new_crypto_df.head()


# In[10]:


# Create a new DataFrame that holds only the cryptocurrencies names.
crypto_names_df = pd.DataFrame(new_crypto_df['CoinName'])
crypto_names_df.head()


# In[11]:


# Drop the 'CoinName' column since it's not going to be used on the clustering algorithm.
new_crypto_df.drop(columns=["CoinName"], inplace=True)
new_crypto_df.head()


# In[12]:


#new_crypto_df.drop(columns=["Unnamed: 0"], inplace=True)
#new_crypto_df.head()


# In[13]:


# Use get_dummies() to create variables for text features.
import pandas as pd
from pathlib import Path

X = pd.get_dummies(new_crypto_df, columns=["Algorithm", "ProofType"])

#X = pd.get_dummies(new2_crypto_df)

X.head()


# In[15]:


print(X.dtypes)


# In[17]:


X['Unnamed: 0'].astype(float)


# In[18]:


#standardize the dataset using StandardScaler()
#object = StandardScaler()
#object.fit_transform(X) 




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

standardized_data = scaler.fit_transform(X)

print(standardized_data)


# ### Deliverable 2: Reducing Data Dimensions Using PCA

# In[ ]:


# Using PCA to reduce dimension to three principal components.
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(X)
data_pca = pca.transform(X)
data_pca = pd.DataFrame(X,columns=['PC1','PC2','PC2'])
data_pca.head()


# In[ ]:


# Create a DataFrame with the three principal components.


# ### Deliverable 3: Clustering Crytocurrencies Using K-Means
# 
# #### Finding the Best Value for `k` Using the Elbow Curve

# In[ ]:


# Create an elbow curve to find the best value for K.
# YOUR CODE HERE


# Running K-Means with `k=4`

# In[ ]:


# Initialize the K-Means model.
# YOUR CODE HERE

# Fit the model
# YOUR CODE HERE

# Predict clusters
# YOUR CODE HERE


# In[ ]:


# Create a new DataFrame including predicted clusters and cryptocurrencies features.
# Concatentate the crypto_df and pcs_df DataFrames on the same columns.
# YOUR CODE HERE

#  Add a new column, "CoinName" to the clustered_df DataFrame that holds the names of the cryptocurrencies. 
# YOUR CODE HERE

#  Add a new column, "Class" to the clustered_df DataFrame that holds the predictions.
# YOUR CODE HERE

# Print the shape of the clustered_df
print(clustered_df.shape)
clustered_df.head(10)


# ### Deliverable 4: Visualizing Cryptocurrencies Results
# 
# #### 3D-Scatter with Clusters

# In[ ]:


# Creating a 3D-Scatter with the PCA data and the clusters
# YOUR CODE HERE


# In[ ]:


# Create a table with tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Print the total number of tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Scaling data to create the scatter plot with tradable cryptocurrencies.
# YOUR CODE HERE


# In[ ]:


# Create a new DataFrame that has the scaled data with the clustered_df DataFrame index.
# YOUR CODE HERE

# Add the "CoinName" column from the clustered_df DataFrame to the new DataFrame.
# YOUR CODE HERE

# Add the "Class" column from the clustered_df DataFrame to the new DataFrame. 
# YOUR CODE HERE

plot_df.head(10)


# In[ ]:


# Create a hvplot.scatter plot using x="TotalCoinsMined" and y="TotalCoinSupply".
# YOUR CODE HERE


# In[ ]:




