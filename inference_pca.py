import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA

import open3d as o3d
from utils import *
from data_utils.collations import numpy_to_sparse_tensor, SparseCollation
from data_utils.data_map import color_map, labels, labels_poss, content
from data_utils.ioueval import iouEval

def set_device(args):
    if args.use_cuda:
        device = torch.device("cuda")
        torch.cuda.set_device(args.device_id)
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def run_inference(model, args):
    data_val = data_loaders[args.dataset_name](
        root=args.data_dir,
        split='train',
        intensity_channel=args.use_intensity,
        pre_training=False,
        pca=True,
        percentage=args.percent,
        resolution=args.sparse_resolution
    )

    val_loader = torch.utils.data.DataLoader(
        data_val,
        batch_size=args.batch_size,
        collate_fn=SparseCollation(args.sparse_resolution, np.inf),
        shuffle=False,
    )

    pcd_feats, pcd_coords, pcd_file_paths = [], [], []

    for x_coord, x_feats, x_label, d_path in tqdm(val_loader):
        x, y = numpy_to_sparse_tensor(x_coord, x_feats, x_label)

        if 'UNet' in args.sparse_model:
            y = y[:, 0]
        else:
            y = torch.from_numpy(np.asarray(y))[:, 0]

        h = model(x)
        pcd_feats.append(h.F.cpu().detach())
        pcd_coords.append(x.C.cpu().detach())
        pcd_file_paths.extend(list(d_path))

    print(f"Processed {len(pcd_feats)} point clouds.")

    return pcd_feats, pcd_coords, pcd_file_paths

def apply_pca_and_save(pcd_feats, pcd_coords, paths, sparse_resolution, save_path):
    all_feats = torch.cat(pcd_feats, dim=0)
    print("Running PCA on shape:", all_feats.shape)

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(all_feats.numpy())

    print("Explained variance:", pca.explained_variance_ratio_)
    print("Total variance explained:", np.sum(pca.explained_variance_ratio_))

    keys = list(range(len(pcd_feats)))
    lengths = [f.shape[0] for f in pcd_feats]

    start = 0
    pca_dict = {}
    for i, key in enumerate(keys):
        end = start + lengths[i]
        pca_tensor = torch.tensor(pca_result[start:end], dtype=torch.float32)
        pca_dict[i] = {
            'feat': pca_tensor,
            'coords': pcd_coords[i]
        }
        start = end

    pca_dict['paths'] = paths
    pca_dict['sparse_resolution'] = sparse_resolution
    torch.save(pca_dict, save_path)
    print(f"PCA-reduced features saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Inference + PCA pipeline")
    parser.add_argument('--dataset-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--checkpoint', default='tarl', type=str)
    parser.add_argument('--best', type=str, default='epoch14')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet')
    parser.add_argument('--sparse-resolution', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--use-cuda', action='store_false', default=True)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--use-intensity', action='store_false', default=True)
    parser.add_argument('--percent', type=float, default=0.1)
    args = parser.parse_args()

    set_deterministic()
    device = set_device(args)

    resnet = get_model(args, torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor)
    model_path = Path(args.log_dir) / f"{args.best}_model_{args.checkpoint}.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit()

    ckpt = torch.load(model_path, map_location=device)
    resnet.load_state_dict(ckpt['model'])
    print(f"Loaded model from {model_path}, epoch {ckpt['epoch']}")
    resnet = resnet.to(device).eval()

    feats, coords, paths = run_inference(resnet, args)

    # # Save original features
    # original_save_path = Path(args.log_dir) / f"pcd_feats{100.0 * args.percent}.pt"
    # original_dict = {
    #     i: {'feat': feats[i], 'coords': coords[i]} for i in range(len(feats))
    # }
    # original_dict['paths'] = paths
    # original_dict['sparse_resolution'] = args.sparse_resolution
    # torch.save(original_dict, original_save_path)
    # print(f"Original features saved to: {original_save_path}")

    # Save PCA-reduced features
    pca_save_path = Path(args.log_dir) / f"pcd_feats{100.0 * args.percent}_pca.pt"
    apply_pca_and_save(feats, coords, paths, args.sparse_resolution, pca_save_path)


#python inference_with_pca.py --dataset-name SemanticKITTI --data-dir ./Datasets/SemanticKITTI/ --log-dir checkpoint/downstream_task/percent_100.0/TARL 
# --checkpoint tarl --sparse-model MinkUNet --sparse-resolution 0.05 --best epoch14 --use-cuda --use-intensity  --percent 1.0

if __name__ == "__main__":
    main()
