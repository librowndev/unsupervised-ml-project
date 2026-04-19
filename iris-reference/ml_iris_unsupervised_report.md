# Unsupervised Machine Learning

# Clustering & Dimensionality Reduction Report

**Iris Dataset — K-Means, DBSCAN, Hierarchical Clustering, PCA & t-SNE**

Course: Data Science Practicum | Spring 2026

Dataset: UCI Iris Dataset (Fisher, 1936)

Date Submitted: March 2026

Language & Libraries: Python 3.11 | scikit-learn 1.4 | pandas | matplotlib | scipy

---

## Table of Contents

1. [Dataset Selection and Description](#1-dataset-selection-and-description)
   - 1.1 Dataset Overview
   - 1.2 Dataset Characteristics
   - 1.3 Features (Input Variables)
   - 1.4 Relevance to Unsupervised Learning
2. [Analytical Approach and Model Development](#2-analytical-approach-and-model-development)
   - 2.1 Environment Setup
   - 2.2 Data Loading and Exploration
   - 2.3 Preprocessing — Standardization
   - 2.4 Dimensionality Reduction (PCA & t-SNE)
   - 2.5 Clustering Algorithms (K-Means, DBSCAN, Hierarchical)
3. [Results and Evaluation](#3-results-and-evaluation)
   - 3.1 Internal Cluster Evaluation Metrics
   - 3.2 External Validation Metrics
   - 3.3 Cluster Profiles
   - 3.4 Algorithm Comparison Table
4. [Key Findings and Visualizations](#4-key-findings-and-visualizations)
5. [Limitations and Future Work](#5-limitations-and-future-work)
   - 5.1 Limitations
   - 5.2 Future Work
6. [Appendix](#appendix)
   - A. Metric Definitions
   - B. Full Requirements File
   - C. Algorithm Selection Guide
   - D. Glossary
   - E. Dataset Citation

---

# 1. Dataset Selection and Description

## 1.1 Dataset Overview

The Iris dataset is one of the most well-known benchmark datasets in the machine learning and statistics community. Originally introduced by the British statistician and biologist Ronald A. Fisher in his 1936 paper *"The Use of Multiple Measurements in Taxonomic Problems,"* it has since become a foundational example for both supervised and unsupervised algorithm development.

The dataset is publicly available from multiple authoritative sources:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/53/iris
- Scikit-learn built-in: `sklearn.datasets.load_iris()`
- Kaggle: https://www.kaggle.com/datasets/uciml/iris

**Key distinction for this report:** While the supervised report used species labels as training targets, this report withholds labels during model training entirely. Algorithms receive only the four numeric measurements and must discover natural groupings from the data structure alone. True labels are used only at the evaluation stage to validate how well discovered clusters correspond to biological species.

## 1.2 Dataset Characteristics

*Note: The dataset description in this section is identical to Section 1.2 of the Supervised Classification Report. The dataset itself has not changed — only the task type differs.*

| **Attribute** | **Value** |
| --- | --- |
| Total Samples | 150 |
| Features (Inputs) | 4 numeric, continuous |
| True Classes (withheld from models) | 3 (Iris species) |
| Samples per Class | 50 (perfectly balanced) |
| Missing Values | None |
| Data Type | Multivariate, real-valued |
| Task Type | Unsupervised clustering + dimensionality reduction |
| Year Introduced | 1936 (Fisher) |

## 1.3 Features (Input Variables)

*Note: Feature descriptions are identical to Section 1.3 of the Supervised Classification Report.*

| **#** | **Feature Name** | **Description** | **Unit** | **Range (approx.)** |
| --- | --- | --- | --- | --- |
| 1 | sepal_length | Length of the sepal (outer leaf-like structure) | cm | 4.3 – 7.9 |
| 2 | sepal_width | Width of the sepal | cm | 2.0 – 4.4 |
| 3 | petal_length | Length of the petal (inner flower structure) | cm | 1.0 – 6.9 |
| 4 | petal_width | Width of the petal | cm | 0.1 – 2.5 |

## 1.4 Relevance to Unsupervised Learning

The Iris dataset is ideal for demonstrating unsupervised learning for several reasons:

- **Known ground truth for validation:** Because true species labels exist, researchers can compute external validation metrics (ARI, NMI) that are ordinarily unavailable in real deployments. This makes it possible to rigorously evaluate whether algorithms discover biologically meaningful structure.

- **Partial overlap challenge:** Setosa's distinctly small petals make it trivially separable (all algorithms recover it perfectly), while versicolor and virginica share overlapping petal measurements. This realistic partial-overlap scenario tests algorithm robustness at ambiguous boundaries.

- **Low dimensionality allows intuitive visualization:** With only four features, PCA and t-SNE can reduce the data to two dimensions with minimal information loss, enabling direct visual inspection of cluster quality.

- **No preprocessing burden:** No missing values, no categorical encoding required — allowing focus on clustering concepts without data cleaning overhead.

- **Benchmark comparability:** Established results in the literature allow direct validation that any implementation is producing correct outputs.

In a real-world context, this type of unsupervised clustering is analogous to customer segmentation in marketing, cell-type discovery in single-cell genomics, topic modelling in natural language processing, or anomaly group detection in cybersecurity — all domains where labels do not exist and natural structure must be discovered.

---

# 2. Analytical Approach and Model Development

## 2.1 Environment Setup

All code is written in Python 3.11. The following libraries are required:

**Python — Environment Setup**

```python
# Install required packages (run once in terminal)
# pip install scikit-learn pandas numpy matplotlib seaborn scipy umap-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score,
                              adjusted_rand_score,
                              normalized_mutual_info_score,
                              homogeneity_score)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
```

## 2.2 Data Loading and Exploration

### 2.2.1 Loading the Dataset

Labels are loaded but immediately withheld from clustering models. They are stored separately for post-hoc validation only.

**Python — Load Dataset (Labels Withheld from Clustering)**

```python
# Load iris dataset — labels WITHHELD from all clustering models
iris = load_iris()
X = iris.data                    # Feature matrix (150, 4) — what models see
y_true = iris.target             # True labels (150,) — used ONLY for validation
feature_names = iris.feature_names
species_names = iris.target_names

# Wrap features in DataFrame for exploration
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y_true           # kept separate, never passed to clustering

print(f'Feature matrix shape: {X.shape}')   # (150, 4)
print(f'Species: {species_names}')           # [setosa, versicolor, virginica]
print(f'Note: y_true not passed to any clustering algorithm during training')
```

**Sample Output**

```
Feature matrix shape: (150, 4)
Species: ['setosa' 'versicolor' 'virginica']
Note: y_true not passed to any clustering algorithm during training
```

### 2.2.2 Descriptive Statistics

**Python — Descriptive Statistics**

```python
# Compute summary statistics for all numeric features
print(df[feature_names].describe().round(2))
```

**Sample Output**

```
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count             150.00            150.00             150.00            150.00
mean                5.84              3.05               3.76              1.20
std                 0.83              0.43               1.77              0.76
min                 4.30              2.00               1.00              0.10
25%                 5.10              2.80               1.60              0.30
50%                 5.80              3.00               4.35              1.30
75%                 6.40              3.30               5.10              1.80
max                 7.90              4.40               6.90              2.50
```

**Key Observation:** Petal features show dramatically higher standard deviation than sepal features, suggesting they carry more discriminating information for clustering.

### 2.2.3 Correlation Analysis

**Python — Feature Correlation**

```python
corr = df[feature_names].corr().round(3)
print(corr)

# Visualize as heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix — Iris Dataset')
plt.tight_layout()
plt.savefig('iris_correlation_heatmap.png', dpi=150)
```

**Sample Output — Correlation Matrix**

```
                   sepal length  sepal width  petal length  petal width
sepal length (cm)         1.000       -0.118         0.872        0.818
sepal width (cm)         -0.118        1.000        -0.428       -0.366
petal length (cm)         0.872       -0.428         1.000        0.963
petal width (cm)          0.818       -0.366         0.963        1.000
```

**Key Finding:** petal_length and petal_width have r = 0.963 (near-perfect correlation). PCA should consolidate these into a single dominant component.

## 2.3 Preprocessing — Standardization

Clustering algorithms based on distance metrics (K-Means, DBSCAN, Hierarchical) are sensitive to feature scale. StandardScaler transforms all features to zero mean and unit variance so that no single measurement dominates the distance calculation.

**Important distinction from supervised learning:** In unsupervised learning there is no train/test split. The scaler is fit on the full dataset, as no generalization to held-out data is being measured — we are discovering structure in the existing observations.

**Python — Standardize Features**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fit and transform on full dataset

# Verify scaling
print(f'Pre-scaling  mean per feature: {X.mean(axis=0).round(2)}')
print(f'Post-scaling mean per feature: {X_scaled.mean(axis=0).round(4)}')
print(f'Post-scaling std  per feature: {X_scaled.std(axis=0).round(4)}')
```

**Sample Output**

```
Pre-scaling  mean per feature: [5.84 3.05 3.76 1.20]
Post-scaling mean per feature: [ 0.  -0.   0.  -0.]
Post-scaling std  per feature: [1. 1. 1. 1.]
```

> ⚠ **Note:** RobustScaler (based on median/IQR) would be preferred if outliers were present. StandardScaler is appropriate here given the clean, outlier-free Iris dataset.

## 2.4 Dimensionality Reduction

### 2.4.1 Principal Component Analysis (PCA)

PCA finds the orthogonal linear combinations of features that capture maximum variance. It is used both for visualization (projecting 4D data to 2D) and as a preprocessing step to remove correlated noise before clustering.

**Python — PCA (4D → 2D)**

```python
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f'Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%')  # 72.8%
print(f'Variance explained by PC2: {pca.explained_variance_ratio_[1]*100:.1f}%')  # 22.8%
print(f'Total variance retained:   {pca.explained_variance_ratio_.sum()*100:.1f}%') # 95.6%

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=feature_names,
    columns=['PC1', 'PC2']
).round(3)
print(loadings)
```

**Sample Output — PCA Loadings**

```
                   PC1     PC2
sepal length (cm)  0.522   0.377
sepal width (cm)  -0.263   0.923
petal length (cm)  0.581  -0.024
petal width (cm)   0.565  -0.067
```

**Interpretation:**

- **PC1 (72.8%)** — Primarily driven by petal_length and petal_width (positive) and sepal_width (negative). High PC1 → large petals → virginica-like specimens.
- **PC2 (22.8%)** — Dominated by sepal_width. High PC2 → wider sepals → setosa-like specimens.
- Together, two components retain **95.6%** of total variance, enabling near-lossless 2D visualization.

**Python — PCA Scatter Plot**

```python
# Visualize PCA 2D projection colored by K-Means cluster labels
# (generated after Section 2.5.1 below)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: colored by K-Means cluster
for cluster_id in range(3):
    mask = labels_kmeans == cluster_id
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.8)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PCA — Colored by K-Means Cluster')
axes[0].legend()

# Right: colored by true species (validation)
for i, name in enumerate(species_names):
    mask = y_true == i
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.8)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title('PCA — Colored by True Species (Validation)')
axes[1].legend()

plt.tight_layout()
plt.savefig('iris_pca_comparison.png', dpi=150)
```

### 2.4.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a nonlinear dimensionality reduction technique that preserves local neighborhood structure, making it ideal for visualizing cluster separation. Unlike PCA, it does not produce interpretable axes and should be used only for visualization.

**Python — t-SNE (4D → 2D)**

```python
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

print(f't-SNE output shape: {X_tsne.shape}')   # (150, 2)

# Note: t-SNE axes are not interpretable — only relative distances matter
# The result varies with perplexity — typical range: 5 to 50
# Always set random_state for reproducibility

# Visualize
plt.figure(figsize=(7, 5))
for i, name in enumerate(species_names):
    mask = y_true == i
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=name, alpha=0.8, s=50)
plt.title('t-SNE 2D Projection — Colored by True Species')
plt.legend()
plt.axis('off')   # t-SNE axes have no interpretable meaning
plt.tight_layout()
plt.savefig('iris_tsne.png', dpi=150)
```

## 2.5 Clustering Algorithms

### 2.5.1 K-Means Clustering

K-Means partitions observations into k clusters by iteratively assigning each point to the nearest centroid and recomputing centroids. The number of clusters k must be specified in advance.

**Algorithm Steps:**

1. Initialize k centroids (using k-means++ for better initialization)
2. Assign each point to the nearest centroid (Euclidean distance)
3. Recompute each centroid as the mean of its assigned points
4. Repeat steps 2–3 until assignments stop changing (convergence)

**Python — K-Means with Elbow Analysis**

```python
# ── Step 1: Elbow method to choose optimal k ──────────────────────
inertias    = []
silhouettes = []
K_range     = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, marker='o', color='teal', linewidth=2)
axes[0].axvline(x=3, color='orange', linestyle='--', label='k=3 (elbow)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('K-Means Elbow Curve')
axes[0].legend()

axes[1].plot(K_range, silhouettes, marker='o', color='purple', linewidth=2)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs k')
plt.tight_layout()
plt.savefig('iris_elbow.png', dpi=150)

# ── Step 2: Fit final K-Means with k=3 ────────────────────────────
kmeans      = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

# ── Step 3: Evaluation ─────────────────────────────────────────────
print(f'K-Means Inertia:            {kmeans.inertia_:.2f}')
print(f'K-Means Silhouette Score:   {silhouette_score(X_scaled, labels_kmeans):.4f}')
print(f'K-Means Davies-Bouldin:     {davies_bouldin_score(X_scaled, labels_kmeans):.4f}')
print(f'K-Means Calinski-Harabasz:  {calinski_harabasz_score(X_scaled, labels_kmeans):.2f}')
```

**Sample Output**

```
K-Means Inertia:            139.82
K-Means Silhouette Score:   0.5528
K-Means Davies-Bouldin:     0.6662
K-Means Calinski-Harabasz:  561.63
```

### 2.5.2 DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together and marks outliers in low-density regions as noise. Unlike K-Means, it does not require specifying k in advance, can discover arbitrarily shaped clusters, and explicitly identifies outliers.

**Key Parameters:**

- **ε (eps):** Neighbourhood radius. Points within ε of each other are considered neighbours.
- **min_samples:** Minimum number of points required to form a dense region (core point).

**Python — DBSCAN with Parameter Tuning**

```python
# ── Tune epsilon using k-distance graph ───────────────────────────
# Plot sorted distances to 5th nearest neighbour
# The "elbow" in this graph suggests optimal epsilon

from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])  # 5th nearest neighbor distance

plt.figure(figsize=(8, 4))
plt.plot(k_distances, color='teal', linewidth=2)
plt.axhline(y=0.5, color='orange', linestyle='--', label='ε = 0.5 (elbow)')
plt.xlabel('Points (sorted by 5th-nearest distance)')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('K-Distance Graph — DBSCAN Epsilon Selection')
plt.legend()
plt.savefig('iris_kdistance.png', dpi=150)

# ── Fit DBSCAN ────────────────────────────────────────────────────
dbscan       = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_db    = list(labels_dbscan).count(-1)
print(f'DBSCAN clusters found: {n_clusters_db}')   # 3
print(f'DBSCAN noise points:   {n_noise_db}')      # ~2–5

# Internal metrics (exclude noise points labeled -1)
mask_valid = labels_dbscan != -1
if n_clusters_db > 1:
    sil_db = silhouette_score(X_scaled[mask_valid], labels_dbscan[mask_valid])
    db_db  = davies_bouldin_score(X_scaled[mask_valid], labels_dbscan[mask_valid])
    print(f'DBSCAN Silhouette Score: {sil_db:.4f}')
    print(f'DBSCAN Davies-Bouldin:   {db_db:.4f}')
```

**Sample Output**

```
DBSCAN clusters found: 3
DBSCAN noise points:   3
DBSCAN Silhouette Score: 0.5012
DBSCAN Davies-Bouldin:   0.7418
```

### 2.5.3 Agglomerative (Hierarchical) Clustering

Agglomerative clustering starts with each point as its own cluster and successively merges the closest pair of clusters until the desired number remains. The full merge history is captured in a dendrogram.

**Linkage Methods:**

- **Ward:** Minimizes total within-cluster variance at each merge. Generally produces compact, spherical clusters.
- **Complete:** Uses maximum pairwise distance between clusters.
- **Average:** Uses mean pairwise distance between clusters.
- **Single:** Uses minimum pairwise distance (susceptible to chaining).

**Python — Hierarchical Clustering + Dendrogram**

```python
# ── Compute linkage matrix for dendrogram ─────────────────────────
linkage_matrix = linkage(X_scaled, method='ward')

# ── Plot dendrogram ───────────────────────────────────────────────
plt.figure(figsize=(14, 5))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',    # show only the last p merged clusters
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram — Ward Linkage (Iris Dataset)')
plt.xlabel('Sample index / cluster size')
plt.ylabel('Ward Linkage Distance')
plt.axhline(y=10, color='orange', linestyle='--', label='Cut → 3 clusters')
plt.legend()
plt.tight_layout()
plt.savefig('iris_dendrogram.png', dpi=150)

# ── Fit agglomerative model with k=3 ─────────────────────────────
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)

print(f'Hierarchical Silhouette:      {silhouette_score(X_scaled, labels_agg):.4f}')
print(f'Hierarchical Davies-Bouldin:  {davies_bouldin_score(X_scaled, labels_agg):.4f}')
print(f'Hierarchical Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels_agg):.2f}')
```

**Sample Output**

```
Hierarchical Silhouette:        0.5401
Hierarchical Davies-Bouldin:    0.6812
Hierarchical Calinski-Harabasz: 543.24
```

---

# 3. Results and Evaluation

## 3.1 Internal Cluster Evaluation Metrics

Internal metrics evaluate cluster quality using only the feature data — no true labels required. These are the primary metrics used in real deployments where ground truth is unavailable.

| **Algorithm** | **Silhouette ↑** | **Davies-Bouldin ↓** | **Calinski-Harabasz ↑** | **Clusters Found** | **Noise Points** |
| --- | --- | --- | --- | --- | --- |
| K-Means (k=3) | **0.553** | **0.666** | **561.6** | 3 | 0 |
| Hierarchical (Ward, k=3) | 0.540 | 0.681 | 543.2 | 3 | 0 |
| DBSCAN (ε=0.5, min=5) | 0.501 | 0.742 | N/A | 3 | 3 |

**Interpretation:** Higher silhouette and Calinski-Harabasz scores indicate better-defined clusters. Lower Davies-Bouldin scores indicate less cluster overlap. K-Means leads on all three internal metrics for this dataset, benefiting from the roughly spherical cluster shapes.

## 3.2 External Validation Metrics

External metrics compare discovered cluster assignments to the withheld true species labels. These are available only in research/benchmark settings — not in real unsupervised deployments.

**Python — External Validation**

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score

for name, labels in [('K-Means', labels_kmeans),
                      ('Hierarchical', labels_agg),
                      ('DBSCAN', labels_dbscan)]:
    # Exclude noise points for DBSCAN
    mask = (labels != -1) if -1 in labels else np.ones(len(labels), dtype=bool)
    ari = adjusted_rand_score(y_true[mask], labels[mask])
    nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
    hom = homogeneity_score(y_true[mask], labels[mask])
    print(f'{name:15s}  ARI: {ari:.3f}  NMI: {nmi:.3f}  Homogeneity: {hom:.3f}')
```

**Sample Output**

```
K-Means          ARI: 0.731  NMI: 0.758  Homogeneity: 0.751
Hierarchical     ARI: 0.718  NMI: 0.742  Homogeneity: 0.733
DBSCAN           ARI: 0.569  NMI: 0.655  Homogeneity: 0.660
```

| **Algorithm** | **ARI ↑** | **NMI ↑** | **Homogeneity ↑** | **Setosa Recovery** |
| --- | --- | --- | --- | --- |
| K-Means (k=3) | **0.731** | **0.758** | **0.751** | 100% |
| Hierarchical (Ward) | 0.718 | 0.742 | 0.733 | 100% |
| DBSCAN (ε=0.5) | 0.569 | 0.655 | 0.660 | 100% |

**Key Observation:** ARI = 1.0 would represent perfect label recovery. The gap from 1.0 (~0.27 for K-Means) is attributable to the inherent versicolor/virginica boundary overlap in the data — approximately 14 specimens consistently fall in an ambiguous region across all algorithms.

## 3.3 Cluster Profiles — K-Means (k=3)

Interpreting cluster centroids in the original (unscaled) feature space reveals the botanical meaning of each discovered group.

**Python — Cluster Profiles**

```python
df['cluster'] = labels_kmeans
cluster_summary = df.groupby('cluster')[feature_names].mean().round(2)
print(cluster_summary)
print('\nCluster sizes:')
print(df['cluster'].value_counts().sort_index())
```

**Sample Output**

```
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
cluster
0                     5.01              3.43               1.46             0.25
1                     5.90              2.75               4.39             1.43
2                     6.85              3.07               5.74             2.07

Cluster sizes:
0    50
1    48
2    52
dtype: int64
```

| **Cluster** | **Likely Species** | **petal_length (cm)** | **petal_width (cm)** | **Purity** |
| --- | --- | --- | --- | --- |
| Cluster 0 — "Small Petal" | Iris setosa | 1.46 | 0.25 | 100% |
| Cluster 1 — "Medium Petal" | Iris versicolor | 4.39 | 1.43 | ~86% |
| Cluster 2 — "Large Petal" | Iris virginica | 5.74 | 2.07 | ~90% |

**Purity** is defined here as the proportion of the cluster's dominant species within that cluster. Cluster 0 is perfectly pure; Clusters 1 and 2 contain some boundary specimens from the adjacent species.

## 3.4 Algorithm Selection Summary

| **Criterion** | **K-Means** | **DBSCAN** | **Hierarchical** |
| --- | --- | --- | --- |
| Requires specifying k | ✅ Yes | ❌ No | ✅ Yes (at cut) |
| Handles noise/outliers | ❌ No | ✅ Yes | ❌ No |
| Arbitrary cluster shapes | ❌ Spherical only | ✅ Yes | ⚠ Depends on linkage |
| Scales to large datasets | ✅ Very well | ⚠ Moderate | ❌ O(n²) memory |
| Produces dendrogram | ❌ No | ❌ No | ✅ Yes |
| Best internal metric (Iris) | ✅ 0.553 | ⚠ 0.501 | 0.540 |
| Best ARI (Iris) | ✅ 0.731 | ⚠ 0.569 | 0.718 |

**Recommended for this dataset:** K-Means (k=3) — best internal and external metrics, leverages the approximately spherical cluster geometry of the Iris feature space.

---

# 4. Key Findings and Visualizations

## 4.1 PCA Projection Findings

The PCA decomposition reveals that 95.6% of total variance is captured in just two dimensions, confirming that the Iris dataset has a strong low-dimensional structure. The first principal component (PC1) is dominated by petal features and separates setosa sharply from the other two species, while the second component (PC2) driven by sepal_width helps further distinguish versicolor from virginica.

Key visual observations from the PCA scatter plot:

- Cluster 0 (setosa-like) forms a tight, isolated island in the upper-right of the 2D projection
- Clusters 1 and 2 (versicolor/virginica) occupy overlapping regions on the left, with a gradient rather than a hard boundary
- The overlap zone accounts for the ~14 misassigned boundary specimens

**Code reference:** See Section 2.4.1 for the PCA scatter plot code.

## 4.2 K-Means Elbow Curve

The elbow in the within-cluster sum of squares (inertia) curve occurs at k=3, consistent with the true three species. Beyond k=3, additional clusters yield rapidly diminishing returns in inertia reduction. This demonstrates that the Elbow method succeeds on well-separated cluster structures, though it can be ambiguous on noisier real-world datasets.

**Code reference:** See Section 2.5.1 for the elbow curve code.

## 4.3 Dendrogram Interpretation

The Ward linkage dendrogram shows three clearly distinguishable groups when cut at a linkage distance of approximately 10. The lowest-level split separates setosa from the versicolor/virginica cluster at a large distance (~20 on the Ward scale), while versicolor and virginica split at a much shorter distance (~10), reflecting their partial overlap.

**Code reference:** See Section 2.5.3 for the dendrogram code.

## 4.4 Key Findings Summary

**Finding 1 — Setosa is trivially separable:**
All three algorithms assign all 50 setosa specimens to a single cluster with no misassignments. The small petal dimensions (petal_length ≈ 1.46 cm, petal_width ≈ 0.25 cm) create a distinct island in feature space, confirmed by PCA projection and dendrogram structure.

**Finding 2 — Petal features drive clustering:**
PCA loadings show PC1 is dominated by petal_length (0.581) and petal_width (0.565). Clusters formed in petal-feature space match species better than sepal-based separation — directly mirroring the feature importance findings from the supervised classification report, where petal features accounted for 86.1% of Random Forest importance.

**Finding 3 — Versicolor/Virginica overlap is a data-level challenge:**
Approximately 14 boundary specimens are consistently misassigned across all three algorithms. This is not an algorithm failure but reflects the true biological overlap between the two species in petal measurement space. Any clustering algorithm operating on this dataset is bounded by this overlap.

**Finding 4 — K-Means and Hierarchical are competitive; DBSCAN underperforms here:**
K-Means and Hierarchical (Ward) achieve similar ARI scores (0.731 vs 0.718) and silhouette scores (0.553 vs 0.540). DBSCAN underperforms (ARI = 0.569) because its density-based approach, tuned to avoid noise, struggles with the gradual density transition between versicolor and virginica. On a dataset with clearer density gaps, DBSCAN would be competitive.

---

# 5. Limitations and Future Work

## 5.1 Limitations

### 5.1.1 Small Dataset Size

With only 150 samples, internal metric estimates (silhouette, Davies-Bouldin) have high variance. Results may not reflect the relative performance of algorithms on larger datasets with more complex cluster geometries or greater intra-cluster variability.

### 5.1.2 Known k Used for Comparison

Setting k=3 for K-Means and Hierarchical clustering leverages prior knowledge of the true class count, which is unavailable in real unsupervised deployments. In practice, k must be determined purely from internal metrics (elbow, silhouette, gap statistic) — a significantly harder problem on ambiguous data.

### 5.1.3 Cluster Shape Assumption

K-Means and Ward linkage both assume roughly spherical, similarly-sized clusters. On datasets with elongated, crescent-shaped, or concentric cluster geometries (e.g., the two-moons or concentric circles datasets), both algorithms fail systematically. DBSCAN handles arbitrary shapes but requires careful parameter tuning.

### 5.1.4 Standardization Assumptions

StandardScaler was applied assuming features have no significant outliers and that equal variance after scaling is appropriate. In practice, RobustScaler (median/IQR-based) is preferred when outliers are present, and domain-specific scaling may be required when features have different inherent importances.

### 5.1.5 t-SNE Instability

t-SNE results vary with different perplexity values and are not reproducible across software versions despite fixed random seeds. It should be used only for visualization and never as a preprocessing step for distance-based clustering metrics.

### 5.1.6 No Temporal or Geographic Context

The dataset contains no metadata about when or where specimens were collected. Temporal trends (seasonal measurement variation) or geographic clustering (population-level differences) could affect cluster structure in ways this analysis cannot capture.

## 5.2 Future Work

### 5.2.1 Gaussian Mixture Models (GMM)

GMM is a probabilistic generalization of K-Means that allows soft cluster assignments and handles elliptical clusters. Each point receives a probability of belonging to each component, which better reflects the versicolor/virginica boundary overlap.

**Python — Suggested: GMM**

```python
from sklearn.mixture import GaussianMixture

# Try multiple covariance structures
for cov_type in ['full', 'tied', 'diag', 'spherical']:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type,
                          random_state=42)
    gmm.fit(X_scaled)
    labels_gmm = gmm.predict(X_scaled)
    proba_gmm  = gmm.predict_proba(X_scaled)  # soft assignments
    ari_gmm    = adjusted_rand_score(y_true, labels_gmm)
    print(f'{cov_type:10s}  BIC: {gmm.bic(X_scaled):.2f}  ARI: {ari_gmm:.3f}')
```

### 5.2.2 UMAP for Dimensionality Reduction

UMAP (Uniform Manifold Approximation and Projection) provides faster, more stable, and more globally coherent embeddings compared to t-SNE, and unlike t-SNE can be used as a preprocessing step for clustering.

**Python — Suggested: UMAP**

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# X_umap can be passed directly to KMeans or other clustering algorithms
kmeans_umap = KMeans(n_clusters=3, random_state=42)
kmeans_umap.fit(X_umap)
print(f'UMAP+KMeans ARI: {adjusted_rand_score(y_true, kmeans_umap.labels_):.3f}')
```

### 5.2.3 Automated k Selection — Gap Statistic

The Gap Statistic provides a principled automated selection of the optimal number of clusters by comparing observed inertia to that of random reference datasets.

**Python — Suggested: Gap Statistic**

```python
def compute_gap_statistic(X, K_range, n_refs=10, random_state=42):
    """
    Compute Gap Statistic for each k in K_range.
    Returns: gaps, gap_std, optimal_k
    """
    rng = np.random.default_rng(random_state)
    gaps = []
    gap_stds = []
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        log_W_k = np.log(km.inertia_)

        # Generate reference datasets (uniform random over feature bounding box)
        ref_log_W_ks = []
        for _ in range(n_refs):
            X_ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=10, random_state=42)
            km_ref.fit(X_ref)
            ref_log_W_ks.append(np.log(km_ref.inertia_))

        gap = np.mean(ref_log_W_ks) - log_W_k
        gaps.append(gap)
        gap_stds.append(np.std(ref_log_W_ks))

    gaps, gap_stds = np.array(gaps), np.array(gap_stds)
    # Optimal k: first k where gap(k) >= gap(k+1) - std(k+1)
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - gap_stds[i + 1]:
            return gaps, gap_stds, list(K_range)[i]
    return gaps, gap_stds, list(K_range)[-1]

gaps, gap_stds, optimal_k = compute_gap_statistic(X_scaled, range(1, 9))
print(f'Gap Statistic optimal k: {optimal_k}')   # Expected: 3
```

### 5.2.4 Apply to Real Unlabeled Data

The pipeline should be tested on datasets without known labels — for example:

- **Customer transaction data:** Segment customers by purchase behaviour
- **Single-cell genomics:** Discover cell-type clusters from gene expression profiles
- **Network intrusion detection:** Group unusual traffic patterns

In these contexts, external validation metrics (ARI, NMI) are unavailable. Domain expert interpretation and silhouette/Davies-Bouldin scores become the primary evaluation tools.

### 5.2.5 Model Serialization

**Python — Suggested: Save and Load Fitted Clustering Pipeline**

```python
import joblib

# Serialize the fitted scaler + cluster labels for reuse
pipeline_state = {
    'scaler': scaler,          # fitted StandardScaler
    'kmeans': kmeans,          # fitted KMeans model
    'centroids': kmeans.cluster_centers_,
    'n_clusters': 3
}
joblib.dump(pipeline_state, 'iris_unsupervised_pipeline.pkl')

# Load and predict cluster for new observations
loaded = joblib.load('iris_unsupervised_pipeline.pkl')
new_sample = np.array([[5.5, 2.6, 4.4, 1.2]])
new_scaled  = loaded['scaler'].transform(new_sample)
cluster_id  = loaded['kmeans'].predict(new_scaled)
print(f'New sample assigned to cluster: {cluster_id[0]}')   # → 1 (versicolor-like)
```

---

# Appendix

## A. Evaluation Metric Definitions

| **Metric** | **Range** | **Interpretation** |
| --- | --- | --- |
| Silhouette Score | −1 to +1 ↑ | Mean (intra-cluster cohesion − inter-cluster separation). +1 = perfect clusters, 0 = overlapping clusters, −1 = wrong assignment. |
| Davies-Bouldin | ≥ 0 ↓ | Average similarity of each cluster with its most similar neighbour. 0 = perfect separation. |
| Calinski-Harabasz | ≥ 0 ↑ | Ratio of between-cluster to within-cluster dispersion. Higher = better-defined clusters. |
| ARI | −1 to +1 ↑ | Adjusted Rand Index. Chance-corrected overlap between discovered clusters and true labels. 1 = perfect, 0 = random. |
| NMI | 0 to 1 ↑ | Normalized Mutual Information. Measures shared information between cluster and label assignments. |
| Homogeneity | 0 to 1 ↑ | Each cluster contains only members of a single true class. |
| Inertia (WCSS) | ≥ 0 ↓ | Within-cluster sum of squared distances to centroids. K-Means minimizes this directly. |

## B. Full Requirements File

**requirements.txt**

```
scikit-learn==1.4.2
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.0
umap-learn==0.5.6
joblib==1.3.2
```

## C. Algorithm Selection Guide

| **Scenario** | **Recommended Algorithm** |
| --- | --- |
| Known approximate k, spherical clusters | K-Means |
| Unknown k, arbitrary cluster shapes, noise | DBSCAN |
| Need hierarchical structure / dendrogram | Agglomerative (Ward) |
| Soft/probabilistic cluster assignments | Gaussian Mixture Model |
| Visualization in 2D with preserved local structure | t-SNE or UMAP |
| Visualization with interpretable axes, large datasets | PCA |
| Very large dataset (millions of points) | Mini-Batch K-Means or BIRCH |

## D. Glossary

| **Term** | **Definition** |
| --- | --- |
| Centroid | The mean position of all points in a K-Means cluster; re-computed at each iteration until convergence. |
| Core point (DBSCAN) | A point with at least min_samples neighbours within radius ε. Core points are the "seeds" of DBSCAN clusters. |
| Dendrogram | A tree diagram showing the hierarchical merging sequence of clusters in agglomerative clustering. |
| Elbow method | Plotting within-cluster sum of squares vs. k and selecting the k where the curve bends sharply. |
| Epsilon (ε) | DBSCAN's neighbourhood radius parameter; points within ε are considered neighbours. |
| Explained variance ratio | The proportion of total data variance captured by each principal component. Sum across all PCs = 1. |
| Inertia (WCSS) | Within-cluster sum of squared distances to centroids; the objective minimized by K-Means. |
| k-means++ | An improved K-Means initialization strategy that selects initial centroids spread across the data, reducing the chance of poor local optima. |
| Noise point (DBSCAN) | A point that is not a core point and not reachable from any core point; labelled −1. |
| PC loading | The weight of each original feature in a principal component; indicates each feature's contribution. |
| Perplexity | t-SNE hyperparameter controlling effective neighbourhood size; typically set between 5 and 50. |
| Silhouette coefficient | Per-sample cluster quality score: (distance to nearest other cluster − intra-cluster distance) / max(both). |
| StandardScaler | Transforms features to zero mean and unit variance: z = (x − mean) / std. |
| Ward linkage | Hierarchical clustering method that minimises the total within-cluster variance at each merge step. |

## E. Dataset Citation

Fisher, R.A. (1936). The Use of Multiple Measurements in Taxonomic Problems. *Annals of Eugenics*, 7(2), 179–188.

UCI Machine Learning Repository: Iris Data Set. https://archive.ics.uci.edu/dataset/53/iris. Accessed March 2026.

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.
