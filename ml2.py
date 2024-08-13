import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = r"C:\Users\Yashraj\Downloads\archive (1)\Mall_Customers.csv"
data = pd.read_csv(file_path)

# Step 2: Preprocess the data
# Select relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Optional: Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Apply K-means clustering
# Choose the number of clusters (K)
k = 5  # You can change this value based on your analysis
kmeans = KMeans(n_clusters=k, random_state=0)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Step 4: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(
    features['Annual Income (k$)'],
    features['Spending Score (1-100)'],
    c=data['Cluster'],
    cmap='viridis',
    label='Clusters'
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='red',
    label='Centroids'
)
plt.title('K-means Clustering of Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
