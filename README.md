# Cold-Start Recommendation with SVD, DBSCAN Clustering, and Active Learning

This repository implements an advanced recommender system that addresses the cold-start user problem using matrix factorization (SVD), K-Means clustering, and active learning strategies (Y-change and Error-change). The system enhances cold-start item selection by leveraging latent embeddings and dynamic hybrid scoring.

---

## Project Features

- **Matrix Factorization via SVD (Surprise library)**
- **DBSCAN Clustering** on item latent embeddings
- **Y-change and Error-change** strategies (risk-adjusted)
- **Hybrid score optimization using EM**
- **Evaluation across cold users and ranking sizes**
- **Cluster visualization and parameter tuning via silhouette score**

---

## File Structure

- `implementation.py` — Main script implementing full enhanced pipeline.
- `base_line_code.py` — Original baseline implementation using Surprise + active learning only.
- `evaluation_results.csv` — Evaluation results per strategy/level.
- `optimized_alphas.csv` — Learned alpha weights per cold user.
- `item_clusters_pca.png` — Cluster visualization plot.
- `cluster_silhouette_scores.png` — Optimization diagnostic plot.
- `alpha_boxplot.png` - Boxplot of $\alpha$ values for each variant (risky, moderate, and conservative)
- `alpha_per_user.png` - line plot to visualize $\alpha$ values for each cold user.
- `base_line.slurm` - SLURM script to execute baseline code in High-Processing Compute supporting SLURM resource manager
- `plot_alpha.py` - to plot the $\alpha$ values from optimized_alphas.csv
- `Project Report.pdf` - Report containing information regarding this project.

---

## Requirements

Install dependencies using pip:

```bash
pip install numpy pandas matplotlib scikit-learn surprise
```
---

## How to Get the Dataset

This project uses a cold-start binary feedback dataset derived from a Dutch luxury e-commerce store.

### Download and Extract

Use the following commands to download and extract the dataset (requires `p7zip` or `7z` installed):

- Step 1: Clone the repository
```bash
git clone [https://github.com/rmlaanen/RS-error-based-learning](https://github.com/rmlaanen/RS-error-based-learning.git)
```

- Step 2: Extract it (requires 7-Zip; install via 'sudo apt install p7zip-full' on Ubuntu)
```bash
7z x useritemmatrix.7z
```
You should now have: `useritemmatrix.csv`

- Step 3: Copy the generated CSV file to the location where this repository is cloned.

---

## Execute Baseline

 As data is large, baseline implementation requires large memory or high-performance computing (HPC). Thus, try to execute `base_line_code.py` using `base_line.slurm` SLURM script.
```bash
sbatch base_line.slurm
```


## Execute the new approach
 Despite data being large, clustering reduces the computational requirements. However, it is recommended to execute the code in HPC.
```bash
python implementation.py
```


## Visualize $\alpha$ values

 Execute following to get visualizations related to $\alpha$ values
```bash
python plot_alpha.py
```
