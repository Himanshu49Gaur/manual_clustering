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

---

## **Results & Visualizations**
### SNN-Based Clustering: Results & Analysis

### 1. Leukemia Dataset

### **Final Cluster Labels**
[1 0 1 1 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 1 1 1 1 1 1 0 1 1 1
1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 1 1]

### **Evaluation Metrics**
- **Adjusted Rand Index (ARI):** `0.1858`
- **Normalized Mutual Info (NMI):** `0.1393`
- **Adjusted Mutual Info (AMI):** `0.1300`
- **Fowlkes-Mallows Index (FMI):** `0.6102`
- **Purity Score:** `0.7222`

### **Visualizations**
- Silhouette Plot 
- Cluster Heatmap 
- Hierarchical Clustering Dendrogram 
- Cluster Visualization 

---

### 2. GLIOMA Dataset

### **Final Cluster Labels**
[2 0 2 0 0 2 0 2 0 0 2 0 0 0 0 0 0 0 0 0 0 3 2 1 2 2 2 3 2 2 2 2 2 2 1 1 1
1 3 1 2 1 1 2 3 3 3 3 2 1]

### **Evaluation Metrics**
- **Adjusted Rand Index (ARI):** `0.2755`
- **Normalized Mutual Info (NMI):** `0.4484`
- **Adjusted Mutual Info (AMI):** `0.4016`
- **Fowlkes-Mallows Index (FMI):** `0.4642`
- **Purity Score:** `0.62`

---

### 3. Supervised Validation (GLIOMA Dataset)
- **Cross-Validation Accuracy Scores:** `[0.7, 0.6, 0.8, 0.9, 0.9]`
- **Mean Accuracy:** `0.78`
- **Standard Deviation:** `0.1166`
- **Best Accuracy:** `0.90`

### **Visualizations**
- Silhouette Plot 
- Cluster Heatmap 
- Hierarchical Clustering Dendrogram
- Cluster Visualization 

---

### **Summary of SNN-Based Clustering**
Algorithm 1 demonstrated **moderate clustering performance** on the Leukemia dataset (Purity `0.72`, FMI `0.61`) and **better performance on the GLIOMA dataset** (NMI `0.44`, ARI `0.27`). Supervised validation on GLIOMA confirmed **robust classification potential with a peak accuracy of 90%**. Visualizations (to be included) will provide insights into cluster structure, quality, and separability.

--- 

## Algorithm 2: SIMLR-Based Clustering Results  

This section presents the results of the **SIMLR** algorithm applied to **Dataset 1 (Leukemia)** and **Dataset 2 (GLIOMA)**. Dimensionality reduction using **PCA** was performed prior to clustering.  

---

### **Dataset 1**  

### **Clustering Summary:**  
- **Cluster Sizes:** {Cluster 1: 50, Cluster 0: 22}  

### **Evaluation Metrics:**  
- **Adjusted Rand Index (ARI):** -0.0248  
- **Normalized Mutual Information (NMI):** 0.0087  
- **Silhouette Score:** 0.0445  
- **Purity Score:** 0.6528  

### **Visualizations**  
1. Learned Similarity Matrix (showing relationships learned by SIMLR).  
2. 2D Clustering Plot (PCA-based cluster visualization).  
3. Cluster-wise Feature Variance Heatmap (highlighting feature importance across clusters).  

---

## **Dataset 2**

### **Clustering Summary:**  
- **Cluster Sizes:** {Cluster 1: 36, Cluster 0: 36}  

### **Evaluation Metrics:**  
- **Adjusted Rand Index (ARI):** -0.0122  
- **Normalized Mutual Information (NMI):** 0.0006  
- **Silhouette Score:** 0.0039  
- **Purity Score:** 0.6528  

### **Visualizations**  
1. Learned Similarity Matrix (capturing similarity structure).  
2. 2D Clustering Plot (visualizing cluster separation).  
3. Cluster-wise Feature Variance Heatmap (identifying discriminative features). 

---

### **Observation:**  
- SIMLR performed poorly on both datasets with near-zero **ARI** and **NMI** values, indicating almost no meaningful clustering structure.
- **Purity scores** remained moderate (~0.65) despite low internal cluster validation scores.  
- Visualizations will provide deeper insights into **why clusters failed to form well** and whether **data distribution impacted performance**.  

--- 

## Algorithm 3: SNN-Cliq Clustering Results  

This section presents the results of the **SNN-Cliq** clustering algorithm applied to the **GLIOMA** and **ALLAML** datasets. Hyperparameters were optimized via **grid search** for best clustering performance.  

---

### **Dataset 1 (GLIOMA Dataset)**  

### **Best Hyperparameters:**  
- **k:** 20  
- **min_shared_neighbors:** 12  

### **Evaluation Metrics:**  
- **Silhouette Score:** 0.121  
- **Purity Score:** 0.749  
- **Adjusted Rand Index (ARI):** 0.235  
- **Normalized Mutual Information (NMI):** 0.373  
- **Fowlkes-Mallows Index (FMI):** 0.432  

### **Visualizations**  
1. Cluster Visualization (showing clear separation of clusters identified by SNN-Cliq).  

---

### **Dataset 2 (ALLAML Dataset)**  

### **Best Hyperparameters:**  
- **k:** 60  
- **min_shared_neighbors:** 6  

### **Evaluation Metrics:**  
- **Silhouette Score:** 0.069  
- **Purity Score:** 0.540  
- **Adjusted Rand Index (ARI):** -0.014  
- **Normalized Mutual Information (NMI):** 0.000  
- **Fowlkes-Mallows Index (FMI):** 0.573  

### **Visualizations**  
1. Cluster Visualization (revealing cluster overlaps and misclassifications).  

---

### **Observation:**  
- **GLIOMA dataset:** SNN-Cliq outperformed previous algorithms with **moderate clustering quality** (higher **ARI**, **NMI**, and **FMI**).  
- **ALLAML dataset:** Clustering results were **weak**, with **negative ARI** and **zero NMI**, indicating minimal structure in the data.  
- Visualization will help assess **cluster compactness** and **why performance varied significantly across datasets**.  

---

## Future Enhancements  

To further improve the performance, usability, and scalability of the clustering framework, the following enhancements are planned:  

### **1. Integration with GPU-Accelerated Libraries (e.g., CuPy, PyTorch)**  
Leveraging GPU acceleration will significantly **reduce computation time**, especially for large-scale biological datasets, enabling **real-time clustering and analysis**.  

### **2. Addition of Benchmark Datasets for Standardized Testing**  
Incorporating widely used benchmark datasets will allow **consistent evaluation** of algorithm performance, ensuring **reproducibility and comparability** across studies.  

### **3. Web-Based Interactive Visualization (e.g., Streamlit)**  
Developing a **web-based dashboard** for interactive visualization will help researchers **explore clusters dynamically**, adjust parameters on the fly, and view **real-time clustering results**.  

### **4. Integration with Deep Learning-Based Embedding Methods**  
Combining clustering with **deep representation learning** (e.g., autoencoders, graph neural networks) can improve **feature extraction** and **clustering accuracy** on high-dimensional data.  

### **5. Automated Parameter Tuning for k (Neighbors, Clusters)**  
Implementing **automated hyperparameter optimization** techniques (e.g., Bayesian optimization, genetic algorithms) will eliminate the need for **manual tuning**, saving time and improving results.  

### **6. Support for Multi-Modal Biological Datasets**  
Extending the framework to handle **multi-omics** or **multi-modal datasets** will enable **comprehensive biological insights**, integrating data such as **gene expression, proteomics, and epigenomics**.  

