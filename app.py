#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:14:48 2025

@author: tanchanokkomonnak
"""

# app. py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import pandas as pd

# load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# set the title
st.title("K-means Clustering Visualizer by Thanchanok Komonnak")

# set the page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

# File uploader for dataset
st.sidebar.header('Upload your dataset')
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # If file is uploaded, read it
    data = pd.read_csv(uploaded_file)
    st.write(data.head())  # Show first few rows of the dataset

    # Perform clustering on uploaded dataset
    st.sidebar.header('Select Number of Clusters')
    n_clusters = st.sidebar.slider('Number of clusters', 1, 10, 3)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_

    # Plot the clustering result
    fig, ax = plt.subplots()
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
    ax.set_title(f'K-means Clustering (n_clusters={n_clusters})')
    ax.legend(['Data points', 'Centroids'])
    st.pyplot(fig)

else:
    # Use default example data when no file is uploaded
    X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)
    y_kmeans = loaded_model.predict(X)
    
    # Plotting
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='X')
    ax.set_title('K-means Clustering')
    ax.legend(['Data points', 'Centroids'])
    st.pyplot(fig)




