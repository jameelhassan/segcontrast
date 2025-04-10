import torch
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

# Path to your input file
pt_path = Path('checkpoint/downstream_task/percent_100.0/DINO_means_BS8_lr4_t3-4.5_epoch=159_loss=1.88_model_t/pcd_feats10.0.pt')

# Load data
data = torch.load(pt_path)

# Sort numeric keys and extract features
keys = sorted([k for k in data.keys() if k not in {'paths', 'sparse_resolution'}], key=lambda x: int(x))

feature_list = []
lengths = []

for k in keys:
    tensor = data[k]['feat']  # shape (M, 96)
    feature_list.append(tensor)
    lengths.append(tensor.shape[0])

# Stack all features: shape (N*M, 96)
all_features = torch.cat(feature_list, dim=0)

# Run PCA to 3 dimensions
pca = PCA(n_components=3)
pca_result = pca.fit_transform(all_features.numpy())  # shape (N*M, 3)

# Explained variance (optional log)
explained_variance = pca.explained_variance_ratio_
total_explained = explained_variance.sum()
print("Explained variance by each component:", explained_variance)
print(f"Total variance explained by 3 components: {total_explained:.4f}")

# Reconstruct the dictionary with PCA features
pca_features_dict = {}
start = 0
# for i, length in enumerate(lengths):
#     end = start + length
#     pca_tensor = torch.tensor(pca_result[start:end], dtype=torch.float32)  # shape (M, 3)
#     pca_features_dict[str(i)] = pca_tensor
#     start = end

for i, k in enumerate(keys):
    length = lengths[i]
    end = start + length
    pca_tensor = torch.tensor(pca_result[start:end], dtype=torch.float32)  # shape (M, 3)
    coords = data[k]['coords']  # Assuming coords is a tensor of shape (M, 3) or similar
    pca_features_dict[k] = {
        'feat': pca_tensor,
        'coords': coords
    }
    start = end

# Add paths back
pca_features_dict['paths'] = data['paths']
pca_features_dict['sparse_resolution'] = data['sparse_resolution']

# Save the final result
output_path = pt_path.parent / (pt_path.stem + '_pca.pt')
torch.save(pca_features_dict, output_path)

print(f"PCA-reduced features (with paths) saved to: {output_path}")
