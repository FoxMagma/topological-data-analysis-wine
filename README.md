# Topological Data Analysis + UMAP on Wine Dataset

**Topological Data Analysis (TDA)** and **UMAP embedding** on the classic Wine dataset (178 samples, 13 features, 3 classes).

## What this project demonstrates

- **Persistent homology** (Rips complex) – detects connected components (H0) and loops (H1)
- **UMAP embedding** – preserves topological structure in 2D and 3D
- **Comparison with PCA** – shows why UMAP retains local relationships that PCA loses

## Files

| File | Description |
|------|-------------|
| `tda_umap_script.py` | Complete Python script |
| `persistence_diagram.png` | TDA: birth vs death of topological features |
| `umap_2d.png` | 2D UMAP projection (colored by wine class) |
| `umap_3d.png` | 3D UMAP projection (rotatable) |
| `pca_vs_umap.png` | Side‑by‑side comparison |

## Requirements

```bash
pip install ripser umap-learn matplotlib seaborn scikit-learn pandas persim