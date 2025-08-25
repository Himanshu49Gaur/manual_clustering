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

## **Project Overview**
High-dimensional biological data, such as **gene expression profiles**, often suffers from:
- **Noise and sparsity**
- **Complex inter-sample relationships**
- **Difficulty in defining true clusters**

This project explores three manual clustering approaches that:
- Construct **robust similarity measures**.
- Integrate **graph-based clustering methods**.
- Provide **quantitative and qualitative evaluation metrics**.

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
