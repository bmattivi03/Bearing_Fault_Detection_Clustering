# Bearing Fault Detection using Machine Learning

## Overview

This project implements a comprehensive pipeline for bearing fault detection using vibration signal analysis and unsupervised machine learning techniques. The analysis leverages signal processing, feature extraction, and advanced clustering algorithms to identify and classify different types of bearing faults.

## Project Structure

The notebook is organized into the following sections:

1. **Import Libraries** - Loading required dependencies for signal processing and ML
2. **Data Loading** - Downloading and parsing bearing vibration data from Google Drive
3. **Data Preprocessing** - Signal filtering and noise reduction
4. **Feature Extraction** - Computing time-domain, frequency-domain, and wavelet features
5. **Segmentation & Feature Engineering** - Windowing signals and extracting statistical features
6. **Data Quality & Cleaning** - Handling missing values and outliers
7. **Feature Scaling & Encoding** - Normalizing features and encoding categorical variables
8. **Features Distribution** - Visualizing feature distributions and correlations
9. **Cluster Model** - UMAP dimensionality reduction and clustering analysis

## Dataset

The dataset consists of bearing vibration signals stored in `.mat` files, including:
- **99 signal recordings** from different bearings
- **Multiple fault types**: Normal, Inner race fault, Outer race fault, Ball fault
- **Different bearing types**: 6203, 6205, 6206, 6207
- **Various load levels**: 0, 1, 2, 3
- **Sampling frequency**: 51,200 Hz

The data is automatically downloaded from Google Drive during execution.

## Features Extracted

### Time Domain Features
- RMS (Root Mean Square)
- Peak-to-Peak amplitude
- Crest Factor
- Kurtosis
- Skewness
- Shape Factor
- Impulse Factor
- Clearance Factor
- Energy

### Frequency Domain Features
- Peak frequency
- Mean frequency
- Frequency standard deviation
- Spectral kurtosis
- Frequency center
- RMS frequency

### Wavelet Features
- Wavelet energy
- Wavelet entropy
- Wavelet variance

## Methodology

### Signal Processing Pipeline

1. **Bandpass Filtering**: Butterworth filter (500-15,000 Hz) to remove noise
2. **Segmentation**: Signals divided into overlapping windows (5,120 samples, 50% overlap)
3. **Feature Extraction**: 25+ features computed per window
4. **Scaling**: StandardScaler normalization for ML compatibility

### Clustering Approach

The project employs multiple clustering algorithms for robust fault detection:

- **UMAP (Uniform Manifold Approximation and Projection)**: Dimensionality reduction optimized through grid search (tested 2-20 components)
- **HDBSCAN (Hierarchical Density-Based Spatial Clustering)**: Density-based clustering for irregular cluster shapes
- **Gaussian Mixture Model (GMM)**: Probabilistic clustering with 7 components
- **Spectral Clustering**: Graph-based clustering with nearest neighbors affinity

### Evaluation Metrics

Three complementary metrics assess clustering quality:

- **Silhouette Score** (Higher is better, range: -1 to 1): Measures cluster cohesion vs separation
- **Calinski-Harabasz Index** (Higher is better): Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index** (Lower is better): Average similarity between clusters

## Dependencies

```python
# Core libraries
numpy
pandas
matplotlib
seaborn

# Signal processing
scipy
pywt (PyWavelets)

# Machine learning
scikit-learn
umap-learn
hdbscan

# Utilities
tqdm
gdown
```

## Installation

```bash
pip install numpy pandas matplotlib seaborn scipy pywavelets scikit-learn umap-learn hdbscan tqdm gdown
```

## Usage

1. **Clone the repository** and open the notebook:
   ```bash
   jupyter notebook group_4_project_2.ipynb
   ```

2. **Run all cells sequentially** - The notebook will:
   - Download the dataset automatically from Google Drive
   - Process all 99 bearing signals
   - Extract features and perform clustering
   - Generate visualizations and metrics

3. **Expected runtime**: ~5-10 minutes depending on hardware

## Results

The analysis typically yields the following performance:

### Best Performing Models

**HDBSCAN**
- Strong silhouette score (~0.45-0.50)
- Excellent Calinski-Harabasz index (>4000)
- Low Davies-Bouldin score (~0.4)
- Automatically determines number of clusters

**Gaussian Mixture Model (GMM)**
- Comparable silhouette score
- Good Calinski-Harabasz index (~3000)
- Best Davies-Bouldin score (~0.35)
- Fixed 7 components

**Spectral Clustering**
- Lower silhouette score (~0.30)
- Weaker Calinski-Harabasz index
- Higher Davies-Bouldin score
- Less suitable for this dataset

### Visualizations

The notebook generates:
- **2D UMAP projections** showing cluster separation
- **3D UMAP visualizations** for enhanced spatial understanding
- **Feature distribution plots** for exploratory analysis
- **Correlation heatmaps** identifying redundant features

## Key Findings

- GMM and HDBSCAN demonstrate superior performance for bearing fault detection
- UMAP dimensionality reduction is crucial for revealing cluster structure
- Optimal UMAP dimensions determined through systematic grid search
- Frequency-domain features particularly discriminative for fault types
- Multi-metric evaluation provides robust model selection

## Technical Highlights

- **Automated hyperparameter optimization** for UMAP components
- **Robust signal preprocessing** with Butterworth filtering
- **Comprehensive feature engineering** combining time, frequency, and wavelet domains
- **Multi-algorithm comparison** ensuring result validity
- **Production-ready code** with error handling and progress tracking

## Authors

Group 4 - Data Visualization Course

## License

This project is provided for educational purposes.

## Acknowledgments

- HUST Bearing Dataset for vibration signal data
- UMAP and HDBSCAN developers for excellent dimensionality reduction and clustering tools
- Scikit-learn community for robust ML infrastructure

## Future Work

- Implement supervised learning for fault classification
- Explore deep learning approaches (CNN, LSTM)
- Add real-time fault detection capability
- Extend to multivariate sensor fusion
- Develop explainability tools for cluster interpretations

---

For questions or issues, please refer to the notebook documentation or contact the project maintainers.
