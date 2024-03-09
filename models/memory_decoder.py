import os
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_sep
from pytorch_metric_learning import distances, losses
from extensions.dcd import DCD
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (
    AverageMeter,
    plot_image_output_gt,
    CosineAnnealingWarmRestartsDecayLR,
)
from .base_image_encoder import BaseImageEncoder
from .snnl import BaseSNNLoss, CDSNNL, CDMSL
from .pcn import PCNEncoder, MLPDecoder
from .treegan import RawGenerator
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GPUMemoryModule:
    def __init__(
        self,
        capacity,
        img_embed_dim,
        pc_embed_dim,
        dtype=torch.float32,
        device="cuda:0",
    ):
        self.capacity = capacity
        self.dtype = dtype
        self.device = device
        self.keys = torch.zeros(capacity, img_embed_dim, dtype=dtype, device=device)
        self.values = torch.zeros(capacity, pc_embed_dim, dtype=dtype, device=device)
        self.current_index = 0
        self.used_memory = 0

    def push(self, key, value):
        if self.current_index >= self.capacity:  # Memory full
            self.pop()

        self.keys[self.current_index] = F.normalize(key, dim=0)
        self.values[self.current_index] = value
        self.current_index += 1
        self.used_memory = min(self.capacity, self.used_memory + 1)

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
        Q = F.normalize(Q, dim=1)
        distance = self.compute_distance(Q, self.keys)

        _, top_k_indices = torch.topk(distance, k, dim=1, largest=False)
        # Expand the dimensions of top_k_indices to prepare for gathering
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(
            -1, -1, self.keys.shape[-1]
        )

        k_nearest_keys = torch.gather(
            self.keys.unsqueeze(0).expand(Q.shape[0], -1, -1),
            dim=1,
            index=top_k_indices_expanded,
        )
        k_nearest_values = torch.gather(
            self.values.unsqueeze(0).expand(Q.shape[0], -1, -1),
            dim=1,
            index=top_k_indices_expanded,
        )

        return k_nearest_keys, k_nearest_values

    def clear(self):
        self.current_index = 0
        self.used_memory = 0
        self.keys.zero_()
        self.values.zero_()


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.4):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, Q, K, V):
        Q = self.linear_q(Q)
        K = self.linear_k(K)
        V = self.linear_v(V)

        output, _ = self.mha(Q, K, V)
        return output


class IterativeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, n_attn=4, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_attn = n_attn
        self.mha_list = nn.ModuleList(
            [MultiHeadAttention(embed_dim, num_heads, dropout) for _ in range(n_attn)]
        )

    def forward(self, Q, K, V):
        for i in range(self.n_attn):
            Q = self.mha_list[i](Q, K, V)
        return Q


class MHARawEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, k_nearest=5, n_attn=4, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_attn = n_attn
        self.k_nearest = k_nearest
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.memory = None

    def forward(self, Q):
        K, V = self.memory.find_k_nearest(Q, self.k_nearest)
        Q = Q.unsqueeze(1)
        prior = self.mha(Q, K, V)
        return prior

    def set_memory(self, memory):
        self.memory = memory


class MHAEncoder(nn.Module):
    def __init__(
        self, embed_dim, num_heads, k_nearest=5, n_attn=4, dropout=0.4, percent=0.8
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_attn = n_attn
        self.k_nearest = k_nearest
        self.img_encoder = BaseImageEncoder(latent_dim=embed_dim)
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.memory = None
        out1 = int(2 * embed_dim * percent)
        out2 = 2 * embed_dim - out1
        self.linear1 = nn.Linear(embed_dim, 2 * out1)
        self.linear2 = nn.Linear(embed_dim, 2 * out2)
        self.linear3 = nn.Linear(4 * embed_dim, 2 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        Q = self.img_encoder(x)
        K, V = self.memory.find_k_nearest(Q, self.k_nearest)
        Q = Q.unsqueeze(1)
        prior = self.mha(Q, K, V)
        feat1 = self.linear1(Q)
        feat2 = self.linear2(prior)
        feat1 = feat1.squeeze(1)
        feat2 = feat2.squeeze(1)
        feat = torch.cat([feat1, feat2], dim=1)
        feat = self.activation(feat)
        feat = self.dropout(feat)
        feat = self.linear3(feat)
        return feat, Q.squeeze(1)

    def set_memory(self, memory):
        self.memory = memory


class MemoryModule(pl.LightningModule):
    def __init__(self, args, alpha=80):
        super(MemoryModule, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.alpha = alpha
        self.delta = args.delta
        self.embed_dim = 1024
        self.k_nearest = 5
        self.pc_feat_encoder = PCNEncoder(latent_dim=self.embed_dim)
        self.pc_encoder = PCNEncoder(latent_dim=self.embed_dim * 2)
        self.decoder = MLPDecoder(latent_dim=self.embed_dim * 2)
        self.img_encoder = BaseImageEncoder(latent_dim=self.embed_dim)
        self.mha = MultiHeadAttention(embed_dim=self.embed_dim, num_heads=8)
        self.memory = GPUMemoryModule(
            capacity=10000, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
        )
        self.val_memory = GPUMemoryModule(
            capacity=400, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
        )
        self.pc_feat_encoder.requires_grad_(False)
        self.pc_encoder.requires_grad_(False)
        # self.decoder.requires_grad_(False)
        self.chamfer_loss = ChamferDistanceL1()
        self.chamfer_loss_sep = ChamferDistanceL1_sep()
        self.dcd = DCD(alpha=self.alpha)
        self.mse = nn.MSELoss()
        self.snn_loss = BaseSNNLoss()
        self.cd_snn_loss = CDSNNL(self.chamfer_loss_sep, delta=0.03, gamma=10)
        self.ms_loss = losses.MultiSimilarityLoss(
            distance=distances.LpDistance(normalize_embeddings=False)
        )
        self.args = args
        self.setup_ops()

    def freeze_networks(self):
        self.pc_feat_encoder.eval()
        self.pc_encoder.eval()
        self.decoder.eval()

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_IE = torch.optim.AdamW(
            self.img_encoder.parameters(), lr=self.args.lr
        )
        self.optimizer_MHA = torch.optim.AdamW(self.mha.parameters(), lr=self.args.lr)
        self.optimizer_D = torch.optim.AdamW(self.decoder.parameters(), lr=self.args.lr)

    def _setup_schedulers(self):
        self.scheduler_IE = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_IE,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )
        self.scheduler_MHA = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_MHA,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )
        self.scheduler_D = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_D,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )

    def setup_ops(self, **kwargs):
        self._setup_optimizers(self.args.lr)
        self._setup_schedulers()
        self.dir = os.path.join(self.args.log_dir, self.args.exp)
        self.validation_step_loss = AverageMeter()
        self.train_step_loss = AverageMeter()
        self.best_train_step_loss = 1e10
        self.best_val_step_loss = 1e10
        self.best_val_loss_epoch = -1
        self.train_step = 0
        self.val_step = 0
        self.writer: SummaryWriter = None
        self.test_encodings = []
        self.test_cats = []

    def on_train_epoch_start(self):
        self.memory.clear()
        self.tmem_log = False
        self.vmem_log = False

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_IE.zero_grad(set_to_none=True)
        self.optimizer_MHA.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        Q = self.img_encoder(A)
        K, V = self.memory.find_k_nearest(Q, self.k_nearest)
        Q = Q.unsqueeze(1)
        prior = self.mha(Q, K, V)
        feat = torch.cat([Q, prior], dim=2)
        feat = feat.squeeze(1)
        out_pc = self.decoder(feat)
        pc_enc = self.pc_encoder(B)
        pc_feat = self.pc_feat_encoder(B)
        out_pc = out_pc.to(torch.float32)

        loss_dcd = self.dcd(out_pc, B)
        loss_chamfer = self.chamfer_loss_sep(out_pc, B)
        # loss_latent = self.mse(feat, pc_enc)
        # loss_snn = self.snn_loss(Q.squeeze(1), taxonomy_id)
        # loss_snn = self.cd_snn_loss(Q.squeeze(1), B)
        loss_snn = self.ms_loss(Q.squeeze(1), taxonomy_id)

        loss = loss_dcd * 500 + loss_snn * self.args.lambda_snn
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(
        #     self.img_encoder.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(
        #     self.mha.parameters(), 1.0)

        self.optimizer_IE.step()
        self.optimizer_MHA.step()
        self.optimizer_D.step()

        for i in range(Q.shape[0]):
            if loss_chamfer[i] > self.delta:
                self.memory.push(Q[i].detach(), pc_feat[i].detach())

        loss_chamfer = torch.mean(loss_chamfer)
        loss_chamfer = loss_chamfer.detach() * 1000
        loss_dcd = loss_dcd.detach() * 1000
        # loss_latent = loss_latent.detach() * 1000

        self.train_step_loss.update(loss_chamfer)
        self.log(
            "loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("dcd_loss", loss_dcd, on_step=True, on_epoch=True, logger=True)
        # self.log('loss_latent', loss_latent, on_step=True,
        #          on_epoch=True, logger=True)
        self.log("loss_snn", loss_snn, on_step=True, on_epoch=True, logger=True)
        self.log("tmem", self.memory.used_memory, prog_bar=True, logger=True)
        self.writer.add_scalar("loss", loss_chamfer, self.train_step)
        self.writer.add_scalar("loss_dcd", loss_dcd, self.train_step)
        # self.writer.add_scalar('loss_latent', loss_latent, self.train_step)
        self.writer.add_scalar("loss_snn", loss_snn, self.train_step)
        self.writer.add_scalar("tmem", self.memory.used_memory, self.train_step)
        if self.memory.used_memory == self.memory.capacity and not self.tmem_log:
            cstep = self.train_step % self.args.train_batches
            self.log("tmemstep", cstep, logger=True)
            self.writer.add_scalar("tmemstep", cstep, self.current_epoch)
            self.tmem_log = True
        self.handle_post_train_step(A, B, out_pc)
        return loss_chamfer

    def handle_post_train_step(self, img, gt, out):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, img.shape[0])
            plot_image_output_gt(
                os.path.join(self.dir, "train", f"train_{self.train_step}.png"),
                img[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(),
                out[index].detach().cpu().numpy(),
                gt[index].detach().cpu().numpy(),
            )

        if self.train_step % 100 == 0:
            index = np.random.randint(0, img.shape[0])
            wandb.log(
                {
                    "GT": wandb.Object3D(gt[index].detach().cpu().numpy()),
                    "Output": wandb.Object3D(out[index].detach().cpu().numpy()),
                    "Image": wandb.Image(
                        img[index]
                        .detach()
                        .cpu()
                        .transpose(1, 0)
                        .transpose(1, 2)
                        .numpy()
                    ),
                }
            )
            if self.writer:
                self.writer.add_image(
                    "Image", img[index].detach().cpu().numpy(), self.train_step
                )
                self.writer.add_mesh(
                    "GT",
                    vertices=gt[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.add_mesh(
                    "Output",
                    vertices=out[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.flush()

    def on_train_epoch_end(self):
        self.scheduler_IE.step()
        self.scheduler_MHA.step()
        self.scheduler_D.step()
        self.writer.add_scalar(
            "train_loss_epoch", self.train_step_loss.avg, self.current_epoch
        )
        self.writer.flush()

    def on_validation_epoch_start(self):
        self.val_memory.clear()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_IE.zero_grad(set_to_none=True)
        self.optimizer_MHA.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        Q = self.img_encoder(A)
        K, V = self.val_memory.find_k_nearest(Q, self.k_nearest)
        Q = Q.unsqueeze(1)
        prior = self.mha(Q, K, V)
        feat = torch.cat([Q, prior], dim=2)
        feat = feat.squeeze(1)
        out_pc = self.decoder(feat)
        pc_feat = self.pc_feat_encoder(B)
        out_pc = out_pc.to(torch.float32)

        loss_chamfer = self.chamfer_loss_sep(out_pc, B)

        for i in range(Q.shape[0]):
            if loss_chamfer[i] > self.delta:
                self.val_memory.push(Q[i].detach(), pc_feat[i].detach())

        loss_chamfer = torch.mean(loss_chamfer)
        loss_chamfer = loss_chamfer.detach() * 1000

        self.validation_step_loss.update(loss_chamfer)
        self.log(
            "val_loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("vmem", self.val_memory.used_memory, logger=True)
        self.writer.add_scalar("val_loss", loss_chamfer, self.val_step)
        self.writer.add_scalar("vmem", self.val_memory.used_memory, self.val_step)
        if (
            self.val_memory.used_memory == self.val_memory.capacity
            and not self.vmem_log
        ):
            cstep = self.val_step % self.args.val_batches
            self.log("vmemstep", cstep, logger=True)
            self.writer.add_scalar("vmemstep", cstep, self.current_epoch)
            self.vmem_log = True
        self.val_step += 1
        return loss_chamfer

    def on_validation_epoch_end(self):
        avg_loss = self.validation_step_loss.avg
        if self.train_step_loss.avg < self.best_train_step_loss:
            self.best_train_step_loss = self.train_step_loss.avg
            self.save_model("best_train_loss_model")
        if avg_loss < self.best_val_step_loss:
            self.best_val_step_loss = avg_loss
            self.best_val_loss_epoch = self.current_epoch
            self.save_model("best_val_loss_model")
        self.save_model("last_epoch_model")
        self.writer.add_scalar(
            "val_loss_epoch", self.validation_step_loss.avg, self.current_epoch
        )
        self.writer.add_scalar(
            "lr", self.scheduler_IE.get_last_lr()[0], self.current_epoch
        )
        # self.delta -= 0.000625*self.current_epoch
        # self.delta = max(self.delta, 0.02)
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            "pc_feat_encoder": self.pc_feat_encoder.state_dict(),
            "pc_encoder": self.pc_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mha": self.mha.state_dict(),
            "img_encoder": self.img_encoder.state_dict(),
            "args": self.args,
            "val_loss": self.best_val_step_loss,
            "train_loss": self.best_train_step_loss,
            "epoch": self.current_epoch,
        }
        if ".ckpt" not in path:
            path += ".ckpt"
        path = os.path.join(self.dir, "checkpoints", path)
        torch.save(data, path)
        # wandb.log_artifact(path, type='model')

    def load_model(self, path):
        print("Loading model from ", path)
        data = torch.load(path)
        self.pc_feat_encoder.load_state_dict(data["pc_feat_encoder"])
        self.pc_encoder.load_state_dict(data["pc_encoder"])
        self.decoder.load_state_dict(data["decoder"])
        if "mha" in data:
            self.mha.load_state_dict(data["mha"])
        else:
            print("MHA not Loaded")

        if "img_encoder" in data:
            self.img_encoder.load_state_dict(data["img_encoder"])
        else:
            print("Image Encoder not Loaded")
        print("Model Loaded Successfully")


class MemoryIAModule(MemoryModule):
    def __init__(self, args, alpha=80):
        super(MemoryIAModule, self).__init__(args, alpha)
        self.embed_dim = 1024
        self.mha = IterativeAttention(embed_dim=self.embed_dim, num_heads=8, n_attn=4)
        self.setup_ops()


class BaseImageEncoderModule(pl.LightningModule):
    def __init__(self, args):
        super(BaseImageEncoderModule, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.args = args
        self.embed_dim = 1024
        self.img_encoder = BaseImageEncoder(latent_dim=self.embed_dim)
        self.snn_loss = BaseSNNLoss()
        self.cd_snn_loss = CDSNNL(
            ChamferDistanceL1_sep(), delta=args.delta, gamma=args.gamma
        )
        self.cd_ms_loss = CDMSL(ChamferDistanceL1_sep(), delta=args.delta)
        self.ms_loss = losses.MultiSimilarityLoss(
            distance=distances.LpDistance(normalize_embeddings=False)
        )
        self.setup_ops(args)

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_IE = torch.optim.AdamW(self.img_encoder.parameters(), lr=lr)

    def _setup_schedulers(self):
        self.scheduler_IE = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_IE,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )

    def setup_ops(self, args):
        self._setup_optimizers(args.lr)
        self._setup_schedulers()
        self.dir = os.path.join(args.log_dir, args.exp)
        self.validation_step_loss = AverageMeter()
        self.train_step_loss = AverageMeter()
        self.best_train_step_loss = 1e10
        self.best_val_step_loss = 1e10
        self.best_val_loss_epoch = -1
        self.train_step = 0
        self.val_step = 0
        self.test_encodings = []
        self.test_cats = []
        self.writer: SummaryWriter = None

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_IE.zero_grad(set_to_none=True)

        Q = self.img_encoder(A)
        # loss_snn = self.snn_loss(Q, taxonomy_id)
        # loss_cd_snn = self.cd_snn_loss(Q, B)
        # loss_ms = self.ms_loss(Q, taxonomy_id)
        loss_ms = self.cd_ms_loss(Q, B)

        loss = loss_ms
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.img_encoder.parameters(), 1.0)

        self.optimizer_IE.step()

        # loss_snn = loss_snn.detach()
        # loss_cd_snn = loss_cd_snn.detach()
        loss_ms = loss_ms.detach()

        self.train_step_loss.update(loss.detach())
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('loss_snn', loss_snn, on_step=True,
        #          on_epoch=True, logger=True)
        # self.log('loss_cd_snn', loss_cd_snn,
        #          on_step=True, on_epoch=True, logger=True)
        self.writer.add_scalar("loss", loss, self.train_step)
        # self.writer.add_scalar('loss_snn', loss_snn, self.train_step)
        # self.writer.add_scalar('loss_cd_snn', loss_cd_snn, self.train_step)
        self.train_step += 1

    def on_train_epoch_end(self):
        self.scheduler_IE.step()
        self.writer.add_scalar(
            "train_loss_epoch", self.train_step_loss.avg, self.current_epoch
        )
        self.writer.flush()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_IE.zero_grad(set_to_none=True)

        Q = self.img_encoder(A)
        # loss_snn = self.snn_loss(Q, taxonomy_id)
        # loss_cd_snn = self.cd_snn_loss(Q, B)
        # loss_ms = self.ms_loss(Q, taxonomy_id)
        loss_ms = self.cd_ms_loss(Q, B)
        self.test_encodings.append(Q.cpu().numpy())
        self.test_cats.append(taxonomy_id.cpu().numpy())

        loss = loss_ms

        # loss_snn = loss_snn.detach()
        # loss_cd_snn = loss_cd_snn.detach()
        loss = loss.detach()

        self.validation_step_loss.update(loss)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # self.log('val_loss_snn', loss_snn, on_step=True,
        #          on_epoch=True, logger=True)
        # self.log('val_loss_cd_snn', loss_cd_snn,
        #          on_step=True, on_epoch=True, logger=True)
        self.writer.add_scalar("val_loss", loss, self.val_step)
        # self.writer.add_scalar('val_loss_snn', loss_snn, self.val_step)
        # self.writer.add_scalar('val_loss_cd_snn', loss_cd_snn, self.val_step)
        # self.val_step += 1

    def on_validation_epoch_end(self):
        avg_loss = self.validation_step_loss.avg
        if self.train_step_loss.avg < self.best_train_step_loss:
            self.best_train_step_loss = self.train_step_loss.avg
            self.save_model("best_train_loss_model")
        if avg_loss < self.best_val_step_loss:
            self.best_val_step_loss = avg_loss
            self.best_val_loss_epoch = self.current_epoch
            self.save_model("best_val_loss_model")
        self.save_model("last_epoch_model")
        self.writer.add_scalar(
            "val_loss_epoch", self.validation_step_loss.avg, self.current_epoch
        )
        self.writer.add_scalar(
            "lr", self.scheduler_IE.get_last_lr()[0], self.current_epoch
        )
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()
        self.plot_tsne()

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            "img_encoder": self.img_encoder.state_dict(),
            "args": self.args,
            "val_loss": self.best_val_step_loss,
            "train_loss": self.best_train_step_loss,
            "epoch": self.current_epoch,
        }
        if ".ckpt" not in path:
            path += ".ckpt"
        path = os.path.join(self.dir, "checkpoints", path)
        torch.save(data, path)

    def load_model(self, path):
        print("Loading model from ", path)
        data = torch.load(path)
        self.img_encoder.load_state_dict(data["img_encoder"])
        print("Model Loaded Successfully")

    def plot_tsne(self):
        if len(self.test_encodings) < 30:
            return
        tsne = TSNE(n_components=2)
        encodings = np.concatenate(self.test_encodings, axis=0)
        cats = np.concatenate(self.test_cats, axis=0)
        tsne_encodings = tsne.fit_transform(encodings)

        # plot the result and save the figure in wandb and tensorboard
        fig, ax = plt.subplots()
        scatter = ax.scatter(tsne_encodings[:, 0], tsne_encodings[:, 1], c=cats)
        # ax.legend(*scatter.legend_elements(), title="Classes")
        ax.grid(True)
        plt.show()
        wandb.log({"t-SNE": wandb.Image(fig)})
        self.writer.add_figure("t-SNE", fig, self.current_epoch)
        self.writer.flush()

        # store the encodings and cats in one file
        with open(os.path.join(self.dir, "encodings.pkl"), "wb") as f:
            pickle.dump([encodings, cats], f)
        self.test_encodings = []
        self.test_cats = []


class MHAEncDecoder(pl.LightningModule):
    def __init__(self, args):
        super(MHAEncDecoder, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.args = args
        self.embed_dim = 1024
        self.pc_feat_encoder = PCNEncoder(latent_dim=self.embed_dim)
        self.pc_encoder = PCNEncoder(latent_dim=self.embed_dim * 2)
        self.decoder = MLPDecoder(latent_dim=self.embed_dim * 2)
        self.mha_encoder = MHAEncoder(
            embed_dim=self.embed_dim, num_heads=8, k_nearest=5, n_attn=4
        )
        self.memory = GPUMemoryModule(
            capacity=4000, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
        )
        self.val_memory = GPUMemoryModule(
            capacity=400, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
        )
        # self.decoder.requires_grad_(False)
        self.pc_feat_encoder.requires_grad_(False)
        self.pc_encoder.requires_grad_(False)
        self.chamfer_loss = ChamferDistanceL1()
        self.chamfer_loss_sep = ChamferDistanceL1_sep()
        self.dcd = DCD(alpha=80)
        self.mse = nn.MSELoss()
        self.snn_loss = BaseSNNLoss()
        self.cd_snn_loss = CDSNNL(self.chamfer_loss_sep, delta=0.03, gamma=10)
        self.ms_loss = losses.MultiSimilarityLoss(
            distance=distances.LpDistance(normalize_embeddings=False)
        )
        self.setup_ops()

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_ME = torch.optim.AdamW(
            self.mha_encoder.parameters(), lr=self.args.lr
        )
        self.optimizer_D = torch.optim.AdamW(self.decoder.parameters(), lr=self.args.lr)

    def _setup_schedulers(self):
        self.scheduler_ME = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_ME,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )
        self.scheduler_D = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_D,
            T_0=self.args.T_0,
            T_mult=2,
            eta_min=5e-8,
            decay_factor=self.args.decay_factor,
        )

    def setup_ops(self, **kwargs):
        self._setup_optimizers(self.args.lr)
        self._setup_schedulers()
        self.dir = os.path.join(self.args.log_dir, self.args.exp)
        self.validation_step_loss = AverageMeter()
        self.train_step_loss = AverageMeter()
        self.best_train_step_loss = 1e10
        self.best_val_step_loss = 1e10
        self.best_val_loss_epoch = -1
        self.train_step = 0
        self.val_step = 0
        self.writer: SummaryWriter = None
        self.test_encodings = []
        self.test_cats = []

    def on_train_epoch_start(self):
        self.memory.clear()
        self.tmem_log = False
        self.vmem_log = False
        self.mha_encoder.set_memory(self.memory)

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_ME.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        feat, Q = self.mha_encoder(A)
        out_pc = self.decoder(feat)
        pc_enc = self.pc_encoder(B)
        pc_feat = self.pc_feat_encoder(B)
        out_pc = out_pc.to(torch.float32)

        loss_dcd = self.dcd(out_pc, B)
        loss_chamfer = self.chamfer_loss_sep(out_pc, B)
        loss_snn = self.ms_loss(Q.squeeze(1), taxonomy_id)

        loss = loss_dcd * 1000 + loss_snn * self.args.lambda_snn
        loss.backward()

        self.optimizer_ME.step()
        self.optimizer_D.step()

        for i in range(Q.shape[0]):
            if loss_chamfer[i] > self.args.delta:
                self.memory.push(Q[i].detach(), pc_feat[i].detach())

        loss_chamfer = torch.mean(loss_chamfer)
        loss_chamfer = loss_chamfer.detach() * 1000
        loss_dcd = loss_dcd.detach() * 1000

        self.train_step_loss.update(loss_chamfer)
        self.log(
            "loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("dcd_loss", loss_dcd, on_step=True, on_epoch=True, logger=True)
        self.log("loss_snn", loss_snn, on_step=True, on_epoch=True, logger=True)
        self.log("tmem", self.memory.used_memory, prog_bar=True, logger=True)
        self.writer.add_scalar("loss", loss_chamfer, self.train_step)
        self.writer.add_scalar("loss_dcd", loss_dcd, self.train_step)
        self.writer.add_scalar("loss_snn", loss_snn, self.train_step)
        self.writer.add_scalar("tmem", self.memory.used_memory, self.train_step)
        if self.memory.used_memory == self.memory.capacity and not self.tmem_log:
            cstep = self.train_step % self.args.train_batches
            self.log("tmemstep", cstep, logger=True)
            self.writer.add_scalar("tmemstep", cstep, self.current_epoch)
            self.tmem_log = True
        self.handle_post_train_step(A, B, out_pc)
        return loss_chamfer

    def handle_post_train_step(self, img, gt, out):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, img.shape[0])
            plot_image_output_gt(
                os.path.join(self.dir, "train", f"train_{self.train_step}.png"),
                img[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(),
                out[index].detach().cpu().numpy(),
                gt[index].detach().cpu().numpy(),
            )

        if self.train_step % 100 == 0:
            index = np.random.randint(0, img.shape[0])
            wandb.log(
                {
                    "GT": wandb.Object3D(gt[index].detach().cpu().numpy()),
                    "Output": wandb.Object3D(out[index].detach().cpu().numpy()),
                    "Image": wandb.Image(
                        img[index]
                        .detach()
                        .cpu()
                        .transpose(1, 0)
                        .transpose(1, 2)
                        .numpy()
                    ),
                }
            )
            if self.writer:
                self.writer.add_image(
                    "Image", img[index].detach().cpu().numpy(), self.train_step
                )
                self.writer.add_mesh(
                    "GT",
                    vertices=gt[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.add_mesh(
                    "Output",
                    vertices=out[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.flush()

    def on_train_epoch_end(self):
        self.scheduler_ME.step()
        self.scheduler_D.step()
        self.writer.add_scalar(
            "train_loss_epoch", self.train_step_loss.avg, self.current_epoch
        )
        self.writer.flush()

    def on_validation_epoch_start(self):
        self.val_memory.clear()
        self.mha_encoder.set_memory(self.val_memory)

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        self.optimizer_ME.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        feat, Q = self.mha_encoder(A)
        out_pc = self.decoder(feat)
        pc_feat = self.pc_feat_encoder(B)
        out_pc = out_pc.to(torch.float32)

        loss_chamfer = self.chamfer_loss_sep(out_pc, B)

        for i in range(Q.shape[0]):
            if loss_chamfer[i] > self.args.delta:
                self.val_memory.push(Q[i].detach(), pc_feat[i].detach())

        loss_chamfer = torch.mean(loss_chamfer)
        loss_chamfer = loss_chamfer.detach() * 1000

        self.validation_step_loss.update(loss_chamfer)
        self.log(
            "val_loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("vmem", self.val_memory.used_memory, logger=True)
        self.writer.add_scalar("val_loss", loss_chamfer, self.val_step)
        self.writer.add_scalar("vmem", self.val_memory.used_memory, self.val_step)
        if (
            self.val_memory.used_memory == self.val_memory.capacity
            and not self.vmem_log
        ):
            cstep = self.val_step % self.args.val_batches
            self.log("vmemstep", cstep, logger=True)
            self.writer.add_scalar("vmemstep", cstep, self.current_epoch)
            self.vmem_log = True
        self.val_step += 1
        return loss_chamfer

    def on_validation_epoch_end(self):
        avg_loss = self.validation_step_loss.avg
        if self.train_step_loss.avg < self.best_train_step_loss:
            self.best_train_step_loss = self.train_step_loss.avg
            self.save_model("best_train_loss_model")
        if avg_loss < self.best_val_step_loss:
            self.best_val_step_loss = avg_loss
            self.best_val_loss_epoch = self.current_epoch
            self.save_model("best_val_loss_model")
        self.save_model("last_epoch_model")
        self.writer.add_scalar(
            "val_loss_epoch", self.validation_step_loss.avg, self.current_epoch
        )
        self.writer.add_scalar(
            "lr", self.scheduler_ME.get_last_lr()[0], self.current_epoch
        )
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            "pc_feat_encoder": self.pc_feat_encoder.state_dict(),
            "pc_encoder": self.pc_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "mha_encoder": self.mha_encoder.state_dict(),
            "args": self.args,
            "val_loss": self.best_val_step_loss,
            "train_loss": self.best_train_step_loss,
            "epoch": self.current_epoch,
        }
        if ".ckpt" not in path:
            path += ".ckpt"
        path = os.path.join(self.dir, "checkpoints", path)
        torch.save(data, path)
        # wandb.log_artifact(path, type='model')

    def load_model(self, path):
        print("Loading model from ", path)
        data = torch.load(path)
        self.pc_feat_encoder.load_state_dict(data["pc_feat_encoder"])
        self.pc_encoder.load_state_dict(data["pc_encoder"])
        self.decoder.load_state_dict(data["decoder"])
        if "mha_encoder" in data:
            self.mha_encoder.load_state_dict(data["mha_encoder"])
        print("Model Loaded Successfully")


class MHAEncTreeDecoder(MHAEncDecoder):
    def __init__(self, args):
        super(MHAEncTreeDecoder, self).__init__(args)
        self.decoder = RawGenerator(
            latent_dim=self.embed_dim * 2, batch_size=args.batch_size
        )
        self.decoder.requires_grad_(False)
        self.setup_ops()
