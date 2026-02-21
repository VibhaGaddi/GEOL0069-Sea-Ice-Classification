# GEOL0069-Sea-Ice-Classification
This project implements an unsupervised machine learning method using Gausian Mixture Models (GMM) to classify Sentinel-3 satellite altimetry data. 
# GEOL0069: Sea Ice and Lead Classification  
## Unsupervised Machine Learning using Sentinel-3 Radar Altimetry

---

📖 **Contents**

- [Overview](#overview)  
- [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)  
- [K-Means Clustering](#k-means-clustering)  
- [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)  
- [Methodology](#methodology)  
- [Implementation Code](#implementation-code)  
- [Results & Visualisation](#results--visualisation)  
- [Repository Structure](#repository-structure)  
- [Requirements](#requirements)  
- [Reference Material](#reference-material)

---

# Overview

This project focuses on the automated classification of **Sea Ice** and **Leads** (open water channels within sea ice) using **unsupervised machine learning**.

Accurate surface classification is essential because:

- Sea ice freeboard calculations depend on correctly identifying water and ice returns  
- Misclassification directly impacts sea ice thickness retrieval  
- Reliable mapping improves understanding of Arctic climate dynamics  

Using **Sentinel-3 radar altimetry waveforms**, this study demonstrates how clustering algorithms can replace manual thresholding with statistically robust, data-driven classification.

---

# Introduction to Unsupervised Learning

Unsupervised learning is a branch of machine learning where algorithms identify patterns in data **without labelled examples**.

In this project, clustering is used to separate:

- Specular radar echoes (Leads)  
- Diffuse radar echoes (Sea Ice)

Two clustering approaches are relevant:

- **K-Means Clustering**  
- **Gaussian Mixture Models (GMM)**

---

# K-Means Clustering

K-Means is a simple unsupervised algorithm that partitions data into **K clusters** by minimizing within-cluster variance. It identifies centroids and assigns each data point to the nearest centroid based on feature similarity.

**Why K-Means?**

- Useful for initial exploratory clustering  
- Fast and scalable  
- Works well when clusters are roughly spherical and separable

**Key Components:**

- **Choosing K** – Number of clusters must be set in advance  
- **Centroid Initialization** – Starting positions affect results  
- **Assignment Step** – Assign points to nearest centroid  
- **Update Step** – Update centroid positions based on cluster members

K-Means yields **hard cluster assignments**, which is useful for simple segmentation but does not include uncertainty information.

---

# Gaussian Mixture Models (GMM)

GMM assumes data are generated from a mixture of Gaussian distributions. Unlike K-Means, it provides **probabilistic cluster assignments** and accounts for covariance between features.

**EM Algorithm Steps:**

- **E-Step**: Compute probability of each point belonging to each Gaussian  
- **M-Step**: Update mean, covariance, and mixing coefficients to maximize likelihood  

GMM is more flexible for radar altimetry features, which can be elongated or overlapping in feature space.

---

# Methodology

The classification uses a **GMM with K = 2 clusters**.

### Features Used

- **Pulse Peakiness (PP)** – Measures sharpness of waveform peaks  
- **Backscatter (σ⁰)** – Measures signal strength of return  
- **Stack Standard Deviation (SSD)** – Quantifies variation across multi-looked waveforms  

Leads produce sharp, high-amplitude peaks due to specular reflection; sea ice produces broader, diffuse returns due to surface roughness.

---

# Implementation Code

## Data Preparation and Clustering

```python
import numpy as np
from sklearn.mixture import GaussianMixture

# Stack features into matrix
X = np.column_stack((peakiness, backscatter, ssd))

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=42)
clusters = gmm.fit_predict(X)

# Identify which cluster corresponds to leads
lead_class = np.argmax(gmm.means_[:, 0])
```

---

## Computing Average Echo Shapes

```python
leads_waves = waveforms[clusters == lead_class]
ice_waves = waveforms[clusters != lead_class]

mean_lead = np.mean(leads_waves, axis=0)
std_lead = np.std(leads_waves, axis=0)

mean_ice = np.mean(ice_waves, axis=0)
std_ice = np.std(ice_waves, axis=0)
```

---

# Results & Visualisation

## 1️⃣ Average Echo Shapes

```python
import matplotlib.pyplot as plt

bins = range(128)

plt.fill_between(bins, mean_lead - std_lead,
                 mean_lead + std_lead, alpha=0.2)
plt.plot(mean_lead, label='Lead (Specular)')

plt.fill_between(bins, mean_ice - std_ice,
                 mean_ice + std_ice, alpha=0.2)
plt.plot(mean_ice, label='Sea Ice (Diffuse)')

plt.xlabel("Waveform Bin")
plt.ylabel("Power")
plt.legend()
plt.show()
```


<img width="572" height="435" alt="Image" src="https://github.com/user-attachments/assets/11ecf47a-a724-4f3a-8972-e9e8cb3e85fb" />

---
## Classification Scatter Plot

The following plot visualises the clustering of radar features using a Gaussian Mixture Model (GMM).  
- `data_cleaned[:,0]` → Backscatter (\(\sigma^0\))  
- `data_cleaned[:,1]` → Pulse Peakiness (PP)  

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned[:, 0], data_cleaned[:, 1], c=clusters, cmap='viridis', s=5)
plt.xlabel("Backscatter (sig_0)")
plt.ylabel("Pulse Peakiness (PP)")
plt.title("GMM Clustering Results: Leads vs Sea Ice")
plt.colorbar(label='Cluster ID')
plt.show()
```
<img width="565" height="433" alt="Image" src="https://github.com/user-attachments/assets/39fbdc5c-9bb7-4586-9e6c-777252d5475b" />
**Figure:** GMM successfully separates leads (specular reflections) from sea ice (diffuse returns) based on physical radar features.

---

##  Confusion Matrix Validation

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Your existing variables
true_labels = flag_cleaned_modified
predicted_gmm = clusters_gmm

# 1. Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# 2. Create the "Blue Hued" Plot
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Sea Ice', 'Leads'], 
            yticklabels=['Sea Ice', 'Leads'])

plt.title('Confusion Matrix: GMM vs ESA Classification')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# 3. Print the text report as well
print("\nClassification Report:")
print(classification_report(true_labels, predicted_gmm))
```

<img width="658" height="547" alt="Image" src="https://github.com/user-attachments/assets/7e8ae749-9962-403c-b7cc-4afb25833cb2" />

---

# Repository Structure

```
Unit_1_Data_Colocating_S2_S3.ipynb
Unsupervised_Learning_Week_4.ipynb
outputs/
```

---

# Requirements & Installation (for Google Colab)

Run the following commands in a Colab notebook cell to install all necessary packages:

```python
# Core scientific libraries
!pip install numpy matplotlib scikit-learn

# Geospatial libraries
!pip install rasterio netCDF4 basemap cartopy
```

- **numpy** – Numerical computing and array handling  
- **matplotlib** – Plotting and visualisation  
- **scikit-learn** – Machine learning algorithms (GMM, K-Means)  
- **rasterio** – Reading and processing raster geospatial data  
- **netCDF4** – Handling Sentinel-3 netCDF data files  
- **basemap / cartopy** – Mapping and geospatial plotting  
#
---

# Reference Material

GEOL0069 AI4EO Unit 2 Practical:  
https://cpomucl.github.io/GEOL0069-AI4EO/Unit_2_Unsupervised_Learning_Methods_updated.html

---

**Author:** [Your Name]  
**Course:** GEOL0069 – AI for Earth Observation  
**Institution:** UCL
