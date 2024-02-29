import os
import pickle

import numpy as np
from pyparsing import C
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from extensions.chamfer_dist import ChamferDistanceL1
from extensions.dcd import DCD
from torch.utils.tensorboard import SummaryWriter
from utils.utils import (AverageMeter, count_parameters, plot_image_output_gt,
                         plot_pcd_one_view, reparameterization, CosineAnnealingWarmRestartsDecayLR)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from .base_image_encoder import BaseImageEncoder
from .snnl import BaseSNNLoss


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)
    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, device="cuda"):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(
            1, 2, self.grid_size ** 2).to(device)  # (1, 2, S)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        # (B,  256, N)
        feature = self.first_conv(xyz.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[
            0]                          # (B,  256, 1)
        # (B,  512, N)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)
        # (B, 1024, N)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[
            0]                           # (B, 1024)

        # decoder
        # (B, num_coarse, 3), coarse point cloud
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)
        # (B, num_coarse, S, 3)
        point_feat = coarse.unsqueeze(
            2).expand(-1, -1, self.grid_size ** 2, -1)
        # (B, 3, num_fine)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(
            B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        # (B, 2, num_fine)
        seed = seed.reshape(B, -1, self.num_dense)

        feature_global = feature_global.unsqueeze(
            2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        # (B, 1024+2+3, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)

        # (B, 3, num_fine), fine point cloud
        fine = self.final_conv(feat) + point_feat

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class PCNEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(PCNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        # encoder
        # (B,  256, N)
        feature = self.first_conv(xyz.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[
            0]                          # (B,  256, 1)
        # (B,  512, N)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)
        # (B, 1024, N)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[
            0]                           # (B, 1024)
        return feature_global


class PCNDecoder(nn.Module):
    def __init__(self, latent_dim=1024, num_dense=16384, grid_size=4, device='cuda'):
        super(PCNDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.grid_size = grid_size
        self.num_coarse = self.num_dense // (self.grid_size ** 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(
            self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(
            1, 2, self.grid_size ** 2).to(device)  # (1, 2, S)

    def forward(self, feature_global):
        B, _ = feature_global.shape
        # decoder
        # (B, num_coarse, 3), coarse point cloud
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)
        # (B, num_coarse, S, 3)
        point_feat = coarse.unsqueeze(
            2).expand(-1, -1, self.grid_size ** 2, -1)
        # (B, 3, num_fine)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(
            B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        # (B, 2, num_fine)
        seed = seed.reshape(B, -1, self.num_dense)

        feature_global = feature_global.unsqueeze(
            2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        # (B, 1024+2+3, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)

        # (B, 3, num_fine), fine point cloud
        fine = self.final_conv(feat) + point_feat

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class PCNEncoderMuVar(nn.Module):
    def __init__(self, latent_dim=1024):
        super(PCNEncoderMuVar, self).__init__()
        self.encoder = PCNEncoder(latent_dim=latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.var = nn.Linear(latent_dim, latent_dim)

    def forward(self, xyz):
        feature = self.encoder(xyz)
        mu = self.mu(feature)
        var = self.var(feature)
        return mu, var


class PCNGenerator(nn.Module):
    def __init__(self, latent_dim=512, num_dense=8192):
        super(PCNGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.image_encoder = BaseImageEncoder(latent_dim=latent_dim)
        self.generator = PCNDecoder(
            latent_dim=latent_dim * 2, num_dense=num_dense)

    def forward(self, img, z):
        img = self.image_encoder(img)
        z = torch.cat((img, z), dim=1)
        _, fine = self.generator(z)
        return fine


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=512, num_dense=8192):
        super(MLPDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.num_dense = num_dense
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_dense)
        )

    def forward(self, z):
        fine = self.mlp(z).reshape(-1, self.num_dense, 3)
        return fine.contiguous()


class PCNModule(pl.LightningModule):
    def __init__(self, args, alpha=80):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.alpha = alpha
        self.encoder = PCNEncoder(latent_dim=args.latent_dim)
        self.decoder = PCNDecoder(latent_dim=args.latent_dim, num_dense=8192)
        self.chamfer_loss = ChamferDistanceL1()
        self.dcd_loss = DCD(alpha=self.alpha)
        self.setup_ops()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_E = torch.optim.RAdam(
            self.encoder.parameters(), lr=self.args.lr)
        self.optimizer_D = torch.optim.RAdam(
            self.decoder.parameters(), lr=self.args.lr)

    def _setup_schedulers(self):
        self.scheduler_E = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_E, T_0=self.args.T_0, T_mult=2, eta_min=5e-8, decay_factor=0.6)
        self.scheduler_D = CosineAnnealingWarmRestartsDecayLR(
            self.optimizer_D, T_0=self.args.T_0, T_mult=2, eta_min=5e-8, decay_factor=0.6)
        self.dir = os.path.join(self.args.log_dir, self.args.exp)

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
        self.lambda_fine = 0.4
        if self.current_epoch > 10:
            self.lambda_fine = 0.5
        if self.current_epoch > 20:
            self.lambda_fine = 0.8
        if self.current_epoch > 30:
            self.lambda_fine = 0.95
        if self.current_epoch > 50:
            self.lambda_fine = 0.99

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        # A = A.to(torch.float32)
        # B = B.to(torch.float32)
        self.optimizer_E.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        encoded = self.encoder(A)
        coarse, fine = self.decoder(encoded)
        loss_dcd_fine = self.dcd_loss(fine, B)
        loss_dcd_coarse = self.dcd_loss(coarse, B)
        loss_chamfer_fine = self.chamfer_loss(fine, B)
        loss = loss_dcd_fine * self.lambda_fine + \
            loss_dcd_coarse * (1 - self.lambda_fine)
        loss.backward()

        self.optimizer_E.step()
        self.optimizer_D.step()

        loss_chamfer_fine *= 1000
        loss_dcd_fine *= 1000
        loss_dcd_coarse *= 1000
        loss_chamfer_fine = loss_chamfer_fine.detach()
        loss_dcd_fine = loss_dcd_fine.detach()
        loss_dcd_coarse = loss_dcd_coarse.detach()

        loss_dict = {
            "loss_dcd_fine": loss_dcd_fine,
            "loss_dcd_coarse": loss_dcd_coarse,
        }
        self.log_dict(loss_dict, on_step=True, on_epoch=True)
        self.log('loss', loss_chamfer_fine, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.train_step_loss.update(loss_chamfer_fine)
        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, self.train_step)
        self.writer.add_scalar('loss', loss_chamfer_fine, self.train_step)
        self.handle_post_train_step(A, coarse, fine)
        return loss

    def handle_post_train_step(self, input, coarse, fine):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, input.size(0))
            plot_pcd_one_view(os.path.join(self.dir, 'train', f"train_{self.train_step}.png"),
                              [input[index].detach().cpu().numpy(), coarse[index].detach().cpu().numpy(), fine[index].detach().cpu().numpy()], ["Input/GT", "Coarse", "Fine"])

        if self.train_step % 1000 == 0:
            index = np.random.randint(0, input.size(0))
            wandb.log({"Input": wandb.Object3D(input[index].detach().cpu().numpy()),
                       "Coarse": wandb.Object3D(coarse[index].detach().cpu().numpy()),
                       "Fine": wandb.Object3D(fine[index].detach().cpu().numpy())})
            if self.writer:
                self.writer.add_mesh('Input', vertices=input[index].detach().cpu().numpy().reshape(1, -1, 3),
                                     global_step=self.train_step)
                self.writer.add_mesh('Coarse', vertices=coarse[index].detach().cpu().numpy().reshape(1, -1, 3),
                                     global_step=self.train_step)
                self.writer.add_mesh('Fine', vertices=fine[index].detach().cpu().numpy().reshape(1, -1, 3),
                                     global_step=self.train_step)
                self.writer.flush()

    def on_train_epoch_end(self):
        self.scheduler_E.step()
        self.scheduler_D.step()
        self.writer.add_scalar('train_loss_epoch', self.train_step_loss.avg,
                               self.current_epoch)
        self.writer.flush()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        encoded = self.encoder(A)
        _, fine = self.decoder(encoded)
        # fine = fine.to(torch.float32)
        loss_chamfer_fine = self.chamfer_loss(fine, B)

        loss_chamfer_fine *= 1000

        self.log('val_loss', loss_chamfer_fine, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_loss.update(loss_chamfer_fine)
        self.writer.add_scalar('val_loss', loss_chamfer_fine, self.val_step)
        self.val_step += 1
        return loss_chamfer_fine

    def on_validation_epoch_end(self):
        avg_loss = self.validation_step_loss.avg
        if self.train_step_loss.avg < self.best_train_step_loss:
            self.best_train_step_loss = self.train_step_loss.avg
            self.save_model('best_train_loss_model')
        if avg_loss < self.best_val_step_loss:
            self.best_val_step_loss = avg_loss
            self.best_val_loss_epoch = self.current_epoch
            self.save_model('best_val_loss_model')
        self.save_model('last_epoch_model')
        self.writer.add_scalar(
            'val_loss_epoch', self.validation_step_loss.avg, self.current_epoch)
        self.writer.add_scalar(
            'lr', self.scheduler_D.get_last_lr()[0], self.current_epoch)
        self.log('lr', self.scheduler_D.get_last_lr()[0])
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()

    def test_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        # A = A.to(torch.float32)
        # B = B.to(torch.float32)
        encoded = self.encoder(A)
        _, fine = self.decoder(encoded)
        # fine = fine.to(torch.float32)
        loss_chamfer_fine = self.chamfer_loss(fine, B)

        loss_chamfer_fine *= 1000

        self.log('test_loss', loss_chamfer_fine, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_loss.update(loss_chamfer_fine)
        self.writer.add_scalar('test_loss', loss_chamfer_fine, self.train_step)
        return loss_chamfer_fine

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'args': self.args,
            'val_loss': self.best_val_step_loss,
            'train_loss': self.best_train_step_loss,
            'epoch': self.current_epoch
        }
        if ".ckpt" not in path:
            path += ".ckpt"
        path = os.path.join(self.dir, 'checkpoints', path)
        torch.save(data, path)
        # wandb.log_artifact(path, type='model')

    def load_model(self, path):
        print("Loading model from ", path)
        data = torch.load(path)
        self.encoder.load_state_dict(data['encoder'])
        self.decoder.load_state_dict(data['decoder'])
        print("Model Loaded Successfully")


class PCNMLPModule(PCNModule):
    def __init__(self, args, alpha=80):
        super().__init__(args)
        # self.args = args
        # self.save_hyperparameters()
        # self.automatic_optimization = False
        # self.alpha = alpha
        self.encoder = PCNEncoder(latent_dim=args.latent_dim)
        self.decoder = MLPDecoder(latent_dim=args.latent_dim, num_dense=8192)
        self.chamfer_loss = ChamferDistanceL1()
        self.dcd_loss = DCD(alpha=self.alpha)
        self.snn_loss = BaseSNNLoss()
        self.setup_ops()

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        # A = A.to(torch.float32)
        # B = B.to(torch.float32)
        self.optimizer_E.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        encoded = self.encoder(A)
        fine = self.decoder(encoded)
        loss_dcd = self.dcd_loss(fine, B)
        loss_chamfer = self.chamfer_loss(fine, B)
        loss_snn = self.snn_loss(encoded, taxonomy_id)
        loss = loss_dcd + loss_snn * self.args.lambda_snn
        loss.backward()

        self.optimizer_E.step()
        self.optimizer_D.step()

        loss_chamfer *= 1000
        loss_dcd *= 1000
        loss_chamfer = loss_chamfer.detach()
        loss_dcd = loss_dcd.detach()

        self.log('loss', loss_chamfer, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('dcd_loss', loss_dcd, on_step=True,
                 on_epoch=True, logger=True)
        self.log('snn_loss', loss_snn, on_step=True,
                 on_epoch=True, logger=True)
        self.train_step_loss.update(loss_chamfer)
        self.writer.add_scalar('dcd_loss', loss_dcd, self.train_step)
        self.writer.add_scalar('loss', loss_chamfer, self.train_step)
        self.writer.add_scalar('snn_loss', loss_snn, self.train_step)
        self.handle_post_train_step(A, fine)
        return loss_dcd

    def handle_post_train_step(self, input, fine):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, input.size(0))
            plot_pcd_one_view(os.path.join(self.dir, 'train', f"train_{self.train_step}.png"),
                              [input[index].detach().cpu().numpy(), fine[index].detach().cpu().numpy()], ["Input/GT", "Fine"])

        save_freq = 1000 * max(1, 16//self.args.batch_size)
        if self.train_step % save_freq == 0:
            index = np.random.randint(0, input.size(0))
            wandb.log({"Input": wandb.Object3D(input[index].detach().cpu().numpy()),
                       "Fine": wandb.Object3D(fine[index].detach().cpu().numpy())})

            if self.writer:
                self.writer.add_mesh('Input', vertices=input[index].detach().cpu().numpy().reshape(1, -1, 3),
                                     global_step=self.train_step)
                self.writer.add_mesh('Fine', vertices=fine[index].detach().cpu().numpy().reshape(1, -1, 3),
                                     global_step=self.train_step)
                self.writer.flush()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        # A = A.to(torch.float32)
        # B = B.to(torch.float32)
        encoded = self.encoder(A)
        fine = self.decoder(encoded)
        self.test_encodings.append(encoded.cpu().numpy())
        self.test_cats.append(taxonomy_id.cpu().numpy())
        # fine = fine.to(torch.float32)
        loss_chamfer_fine = self.chamfer_loss(fine, B)

        loss_chamfer_fine *= 1000

        self.log('val_loss', loss_chamfer_fine, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_loss.update(loss_chamfer_fine)
        self.writer.add_scalar('val_loss', loss_chamfer_fine, self.val_step)
        self.val_step += 1
        return loss_chamfer_fine

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.current_epoch % 10 == 0:
            self.plot_tsne()
        self.test_encodings = []
        self.test_cats = []

    def test_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B) = batch
        # A = A.to(torch.float32)
        # B = B.to(torch.float32)
        encoded = self.encoder(A)
        fine = self.decoder(encoded)
        self.test_encodings.append(encoded.cpu().numpy())
        self.test_cats.append(taxonomy_id.cpu().numpy())
        # fine = fine.to(torch.float32)
        loss_chamfer_fine = self.chamfer_loss(fine, B)

        loss_chamfer_fine *= 1000

        self.log('test_loss', loss_chamfer_fine, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_loss.update(loss_chamfer_fine)
        self.writer.add_scalar('test_loss', loss_chamfer_fine, self.train_step)
        return loss_chamfer_fine

    def plot_tsne(self):
        if len(self.test_encodings) < 30:
            return
        tsne = TSNE(n_components=2)
        encodings = np.concatenate(self.test_encodings, axis=0)
        cats = np.concatenate(self.test_cats, axis=0)
        tsne_encodings = tsne.fit_transform(encodings)

        # plot the result and save the figure in wandb and tensorboard
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            tsne_encodings[:, 0], tsne_encodings[:, 1], c=cats)
        # ax.legend(*scatter.legend_elements(), title="Classes")
        ax.grid(True)
        plt.show()
        wandb.log({"t-SNE": wandb.Image(fig)})
        self.writer.add_figure('t-SNE', fig, self.current_epoch)
        self.writer.flush()

        # store the encodings and cats in one file
        with open(os.path.join(self.dir, 'encodings.pkl'), 'wb') as f:
            pickle.dump([encodings, cats], f)
        self.test_encodings = []
        self.test_cats = []

    def on_test_epoch_end(self):
        self.plot_tsne()
