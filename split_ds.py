from trainer.semantic_kitti_trainer import SemanticKITTITrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from utils import *
import argparse
from numpy import inf
from losses.downstream_criterion import *
import MinkowskiEngine as ME
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SparseSimCLR')

    parser.add_argument('--dataset-name', type=str, default='SemanticKITTI',
                        help='Name of dataset (default: SemanticKITTI')
    parser.add_argument('--data-dir', type=str, default='./Datasets/SemanticKITTI',
                        help='Path to dataset (default: ./Datasets/SemanticKITTI')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input training batch-size')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3')
    parser.add_argument("--decay-lr", default=1e-4, action="store", type=float,
                        help='Learning rate decay (default: 1e-4')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='Tau temperature smoothing (default 0.1)')
    parser.add_argument('--log-dir', type=str, default='checkpoint/downstream_task',
                        help='logging directory (default: checkpoint/downstream_task)')
    parser.add_argument('--checkpoint', type=str, default='classifier_checkpoint',
                        help='model checkpoint (default: classifier_checkpoint)')
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='using cuda (default: True')
    parser.add_argument('--device-id', type=int, default=0,
                        help='GPU device id (default: 0')
    parser.add_argument('--feature-size', type=int, default=128,
                        help='Feature output size (default: 128')
    parser.add_argument('--sparse-resolution', type=float, default=0.05,
                        help='Sparse tensor resolution (default: 0.05')
    parser.add_argument('--percentage-labels', type=float, default=1.0,
                        help='Percentage of labels used for training (default: 1.0')
    parser.add_argument('--num-points', type=int, default=80000,
                        help='Number of points sampled from point clouds (default: 80000')
    parser.add_argument('--sparse-model', type=str, default='MinkUNet',
                        help='Sparse model to be used (default: MinkUNet')
    parser.add_argument('--linear-eval', action='store_true', default=False,
                        help='Fine-tune or linear evaluation (default: False')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='load checkpoint (default: True')
    parser.add_argument('--ckpt-name', type=str, default='lastepoch199_model_tarl',
                        help='name of the checkpoint file. See default')
    parser.add_argument('--use-intensity', action='store_true', default=False,
                        help='use points intensity (default: False')
    parser.add_argument('--contrastive', action='store_true', default=False,
                        help='use contrastive pre-trained weights (default: False')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Number steps to accumulate gradient')
    parser.add_argument('--split-data', action='store_true', default=False,
                        help='Split data into train and validation sets (default: False)')
    parser.add_argument('--seed', type=int, default=0,)

    args = parser.parse_args()

    set_deterministic()
    data_train, data_test = get_dataset(args, pre_training=False)