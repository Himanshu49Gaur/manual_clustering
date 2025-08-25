# **Manual Clustering Algorithms for High-Dimensional Biological Data**

This repository contains three Python implementations of **manual clustering algorithms** specifically designed for **high-dimensional biological datasets**. Each algorithm demonstrates different approaches to uncover underlying structures in complex data, with an emphasis on **interpretability, evaluation, and visualization**.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Algorithms Implemented](#algorithms-implemented)
   - [1. SNN-Based Clustering](#1-snn-based-clustering)
   - [2. SIMLR (Single-cell Interpretation via Multi-kernel Learning Representation)](#2-simlr-single-cell-interpretation-via-multi-kernel-learning-representation)
   - [3. Self-Organizing Maps (SOM) with Spectral Clustering](#3-self-organizing-maps-som-with-spectral-clustering)
5. [Installation & Usage](#installation--usage)
6. [Results & Visualizations](#results--visualizations)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)


---

## **Overview**
Clustering high-dimensional data (e.g., genomic datasets) is a complex task due to:
- **Noise**, **sparsity**, and **non-linear patterns**.
- Standard clustering methods like K-Means often fail to capture intricate structures.
  
This repository explores **three powerful clustering techniques** that overcome these limitations:
- **Shared Nearest Neighbor (SNN)-based Clustering**
- **SIMLR: Multi-kernel Learning for Similarity-based Clustering**
- **Self-Organizing Maps (SOM) with Spectral Clustering**

---

## **Features**
- **Custom Implementations:** Fully manual clustering pipelines (no off-the-shelf clustering functions).
- **Dimensionality Reduction:** PCA, multi-kernel learning, and SOM for visualization.
- **Evaluation Metrics:** Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Silhouette Score, Purity Score.
- **Visual Outputs:** Heatmaps, scatter plots, dendrograms, and learned similarity matrices.

## **Repository Structure**
├── SNN_Clustering.py # Shared Nearest Neighbor clustering implementation

├── SIMLR_Clustering.py # Multi-kernel similarity-based clustering (SIMLR)

├── SOM_Spectral.py # Self-Organizing Maps with Spectral Clustering

└── README.md # Project documentation

---

## **Algorithms Implemented**

### **1. SNN-Based Clustering**
- **Core Idea:** Constructs a similarity graph using **shared nearest neighbors** and detects clusters via **graph connectivity**.
- **Pipeline:**
  1. Standardize data and compute Euclidean distances.
  2. Build a k-nearest neighbor graph.
  3. Compute shared nearest neighbor similarity.
  4. Identify clusters based on dense connectivity.
- **Applications:** Suitable for noisy, high-dimensional datasets like **gene expression profiles**.
- **Visualizations:** Graph visualizations, connectivity heatmaps.

---

### **2. SIMLR (Single-cell Interpretation via Multi-kernel Learning Representation)**
- **Core Idea:** Learns an optimal similarity graph using **multiple Gaussian kernels**, then performs **spectral embedding** for clustering.
- **Pipeline:**
  1. **Manual PCA** for dimensionality reduction.
  2. **Multi-kernel learning:** Constructs similarity graphs using various kernels.
  3. **Network diffusion & spectral embedding:** Refines similarity matrix to reveal structure.
  4. **Manual K-Means:** Clusters samples in low-dimensional space.
- **Datasets:** `ALLAML.mat`, `leukemia.mat`.
- **Evaluation Metrics:** ARI, NMI, Silhouette, Purity.
- **Visualizations:** Heatmap of learned similarity matrix, PCA scatter plots.

---

### **3. Self-Organizing Maps (SOM) with Spectral Clustering**
- **Core Idea:** Uses **SOM** for unsupervised feature mapping and then applies **Spectral Clustering** for final group formation.
- **Pipeline:**
  1. Train SOM to map high-dimensional data to a 2D grid.
  2. Construct a graph from SOM prototypes.
  3. Apply Spectral Clustering to find clusters.
- **Advantages:** Combines **topology-preserving mapping** with **graph-based clustering**.
- **Visualizations:** SOM grid plots, spectral embedding graphs.

---

## **Installation & Usage**

### **Requirements**
- Python 3.8+
- Required libraries :
```
pip install numpy scipy scikit-learn matplotlib seaborn networkx pandas os collections time 

```

---

Clone this repository:
   ```
   git clone https://github.com/Himanshu49Gaur/manual_clustering.git
   cd manual_clustering
   ```
Install dependencies:
   ```
   pip install -r requirements.txt
   ```
