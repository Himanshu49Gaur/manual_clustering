# **Manual Clustering Algorithms for High-Dimensional Biological Data**

This repository contains three Python implementations of **manual clustering algorithms** specifically designed for **high-dimensional biological datasets**. Each algorithm demonstrates different approaches to uncover underlying structures in complex data, with an emphasis on **interpretability, evaluation, and visualization**.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Algorithms Implemented](#algorithms-implemented)
   - [1. Similarity Network Fusion (SNF)](#1-similarity-network-fusion-snf)
   - [2. SIMLR (Single-cell Interpretation via Multi-kernel Learning Representation)](#2-simlr-single-cell-interpretation-via-multi-kernel-learning-representation)
   - [3. Spectral Clustering with Shared Nearest Neighbor (SNN)](#3-spectral-clustering-with-shared-nearest-neighbor-snn)
4. [Installation & Requirements](#installation--requirements)
5. [Usage](#usage)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results & Visualizations](#results--visualizations)
8. [Future Work](#future-work)
9. [License](#license)

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
