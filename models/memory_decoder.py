import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.dcd import DCD
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (AverageMeter, count_parameters, plot_image_output_gt,
                         plot_pcd_one_view, reparameterization)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from .base_image_encoder import BaseImageEncoder
from .snnl import BaseSNNLoss
from .pcn import PCNEncoder, MLPDecoder


class GPUMemoryModule:
    def __init__(self, capacity, img_embed_dim, pc_embed_dim, dtype=torch.float32, device='cuda:0'):
        self.capacity = capacity
        self.dtype = dtype
        self.device = device
        self.keys = torch.zeros(capacity, img_embed_dim,
                                dtype=dtype, device=device)
        self.values = torch.zeros(
            capacity, pc_embed_dim, dtype=dtype, device=device)
        self.current_index = 0

    def push(self, key, value):
        if self.current_index >= self.capacity:  # Memory full
            self.pop()

        self.keys[self.current_index] = key
        self.values[self.current_index] = value
        self.current_index += 1

    def pop(self):
        self.current_index = self.current_index % self.capacity

    def get(self, key):
        index = (self.keys == key).nonzero().squeeze()
        if index.numel() == 0:
            return None  # Key not found
        else:
            return self.values[index]

    def compute_distance(self, Q, K):
        # Calculate the pairwise dot product of Q and K.
        qkt = torch.matmul(Q, K.T)

        # Calculate the norms of Q and K.
        q_norm = torch.norm(Q, dim=1, keepdim=True)  # Transpose q_norm
        k_norm = torch.norm(K, dim=1, keepdim=True).T  # Transpose k_norm

        # Compute the distance using the formula:
        # distance = ||Q||^2 + ||K||^2 - 2 * QKT
        distance = q_norm**2 + k_norm**2 - 2 * qkt
        # distance = (q_norm**2).expand_as(qkt) + (k_norm**2).expand_as(qkt) - 2 * qkt
        return distance

    def find_k_nearest(self, Q, k):
        distance = self.compute_distance(Q, self.keys)

        _, top_k_indices = torch.topk(distance, k, dim=1, largest=False)
        # Expand the dimensions of top_k_indices to prepare for gathering
        top_k_indices_expanded = top_k_indices.unsqueeze(
            -1).expand(-1, -1, self.keys.shape[-1])

        k_nearest_keys = torch.gather(self.keys.unsqueeze(0).expand(
            Q.shape[0], -1, -1), dim=1, index=top_k_indices_expanded)
        k_nearest_values = torch.gather(self.values.unsqueeze(0).expand(
            Q.shape[0], -1, -1), dim=1, index=top_k_indices_expanded)

        return k_nearest_keys, k_nearest_values

    def clear(self):
        self.current_index = 0
        self.keys = torch.zeros(
            self.capacity, self.keys.shape[-1], dtype=self.dtype, device=self.device)
        self.values = torch.zeros(
            self.capacity, self.values.shape[-1], dtype=self.dtype, device=self.device)


class MemoryModule(pl.LightningModule):
    def __init__(self, args, alpha=80):
        super(MemoryModule, self).__init__()
        self.save_hyperparameters(args)
        self.alpha = alpha
        self.img_encoder = BaseImageEncoder(args)
        self.pc_encoder = PCNEncoder(args)
        self.decoder = MLPDecoder(args)
        self.memory = GPUMemoryModule(
            args.memory_capacity, args.img_embed_dim, args.pc_embed_dim)
        self.loss = BaseSNNLoss(temperature=args.temperature)
        self.chamfer_loss = ChamferDistanceL1()
        self.dcd = DCD()
        self.args = args
