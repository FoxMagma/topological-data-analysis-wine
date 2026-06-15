============================================================
# 1. INSTALL REQUIRED LIBRARIES
# ============================================================
!pip install ripser umap-learn matplotlib seaborn scikit-learn pandas persim --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import ripser
from ripser import Rips
import umap
import persim
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. LOAD A REAL DATASET (Wine dataset)
# ============================================================
print("="*60)
print("TOPOLOGICAL DATA ANALYSIS + UMAP VISUALIZATION")
print("="*60)

data = datasets.load_wine()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"\n📊 Dataset: Wine")
print(f"   Samples: {X.shape[0]}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(target_names)} ({', '.join(target_names)})")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3. PERSISTENT HOMOLOGY (TDA) - RIPS COMPLEX
# ============================================================
print("\n" + "="*60)
print("PERSISTENT HOMOLOGY (Rips Complex)")
print("="*60)

# Compute persistence diagrams
rips = Rips(maxdim=2, thresh=10)
diagrams = rips.fit_transform(X_scaled)

# Plot persistence diagram (corrected)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
persim.plot_diagrams(diagrams, title="Persistence Diagram")
plt.xlabel("Birth")
plt.ylabel("Death")

plt.subplot(1, 2, 2)
persim.plot_diagrams(diagrams, lifetime=True, title="Persistence Barcode")
plt.xlabel("Epsilon")

plt.tight_layout()
plt.savefig("persistence_diagram.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: persistence_diagram.png")

# Print summary statistics
h0 = diagrams[0][diagrams[0][:, 1] - diagrams[0][:, 0] > 0.01] if len(diagrams[0]) > 0 else []
h1 = diagrams[1][diagrams[1][:, 1] - diagrams[1][:, 0] > 0.01] if len(diagrams) > 1 and len(diagrams[1]) > 0 else []
h2 = diagrams[2][diagrams[2][:, 1] - diagrams[2][:, 0] > 0.01] if len(diagrams) > 2 and len(diagrams[2]) > 0 else []

print(f"\n📈 Persistence Summary:")
print(f"   H0 features (connected components): {len(h0)}")
print(f"   H1 features (loops): {len(h1)}")
print(f"   H2 features (voids): {len(h2)}")

# ============================================================
# 4. UMAP EMBEDDING (2D AND 3D)
# ============================================================
print("\n" + "="*60)
print("UMAP EMBEDDING (Preserving Topological Structure)")
print("="*60)

# 2D UMAP
reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap_2d = reducer_2d.fit_transform(X_scaled)

# 3D UMAP
reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap_3d = reducer_3d.fit_transform(X_scaled)

# Plot 2D UMAP
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i, class_name in enumerate(target_names):
    mask = y == i
    plt.scatter(X_umap_2d[mask, 0], X_umap_2d[mask, 1], 
                c=colors[i], label=class_name, alpha=0.7, edgecolors='k', s=80)

plt.title("UMAP Projection (2D) - Wine Dataset")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("umap_2d.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: umap_2d.png")

# 3D UMAP
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i, class_name in enumerate(target_names):
    mask = y == i
    ax.scatter(X_umap_3d[mask, 0], X_umap_3d[mask, 1], X_umap_3d[mask, 2],
               c=colors[i], label=class_name, alpha=0.7, s=60)

ax.set_title("UMAP Projection (3D) - Wine Dataset")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")
ax.legend()
plt.tight_layout()
plt.savefig("umap_3d.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: umap_3d.png")

# ============================================================
# 5. COMPARE WITH PCA (Linear) vs UMAP (Topological)
# ============================================================
print("\n" + "="*60)
print("COMPARISON: PCA (Linear) vs UMAP (Topological)")
print("="*60)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, class_name in enumerate(target_names):
    mask = y == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                c=colors[i], label=class_name, alpha=0.7, s=60)
ax1.set_title("PCA (Linear) - Preserves global variance")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.legend()

for i, class_name in enumerate(target_names):
    mask = y == i
    ax2.scatter(X_umap_2d[mask, 0], X_umap_2d[mask, 1], 
                c=colors[i], label=class_name, alpha=0.7, s=60)
ax2.set_title("UMAP (Topological) - Preserves local structure")
ax2.set_xlabel("UMAP 1")
ax2.set_ylabel("UMAP 2")
ax2.legend()

plt.tight_layout()
plt.savefig("pca_vs_umap.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: pca_vs_umap.png")