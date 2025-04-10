import torch
import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from data_utils.data_map import *


def sparse_tensor_to_pcd(coords, pca_feats, sparse_resolution, shift=False):
    """Convert sparse tensor coordinates and PCA features to an Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    points = sparse_resolution * coords
    colors = np.asarray(pca_feats)

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if shift:
        shift_size = (pcd.get_max_bound()[1] - pcd.get_min_bound()[1]) * 1.5
        points[:, 1] += shift_size
        pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def normalize_percentile(feats, lower=1, upper=99):
    """Normalize features using percentiles to avoid influence from outliers."""
    mins = np.percentile(feats, lower, axis=0)
    maxs = np.percentile(feats, upper, axis=0)
    normed = (feats - mins) / (maxs - mins + 1e-8)
    return np.clip(normed, 0, 1)


def save_pca_colored_pointcloud_image_with_labels(ckpt_name, index=0, output_path='pca_vis_labeled.png', shift=False):
    """Load PCA features and visualize the point cloud with RGB color based on PCA."""
    pt_file = Path(f"checkpoint/downstream_task/percent_100.0/{ckpt_name}/pcd_feats1.0_pca.pt")

    if not pt_file.exists():
        raise FileNotFoundError(f"PCA feature file not found at: {pt_file}")

    data = torch.load(pt_file)
    sparse_resolution = data['sparse_resolution']
    pca_features = data[index]['feat'].numpy()
    pcd_coords = data[index]['coords'].numpy()

    # Equalize variance before normalization
    std_devs = pca_features.std(0)
    pca_scaled = pca_features / (std_devs + 1e-8)
    pca_norm = normalize_percentile(pca_scaled)

    pcd_pred = sparse_tensor_to_pcd(pcd_coords[:, 1:], pca_norm, sparse_resolution, shift=shift)
    o3d.visualization.draw_geometries([pcd_pred], window_name=f"{ckpt_name}")

    print(f"Visualized PCA-colored point cloud from checkpoint '{ckpt_name}' (index={index})")


def main():
    parser = argparse.ArgumentParser(description="Visualize PCA-colored LiDAR point cloud from a given checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint folder name under 'checkpoint/downstream_task/percent_100.0/'.")
    parser.add_argument("--index", type=int, default=0, help="Index of the frame to visualize.")
    parser.add_argument("--output_path", type=str, default="pca_vis_labeled.png", help="Output image path (currently unused).")
    parser.add_argument("--shift", action="store_true", help="Whether to shift point cloud along y-axis for visibility.")

    args = parser.parse_args()
    save_pca_colored_pointcloud_image_with_labels(
        ckpt_name=args.ckpt,
        index=args.index,
        output_path=args.output_path,
        shift=args.shift
    )

# python pca_visulaizer.py --ckpt TARL --index 0 --output_path my_vis.png --shift
# DINO_means_BS8_lr4_t3-4.5_epoch=159_loss=1.88_model_t
if __name__ == "__main__":
    main()
