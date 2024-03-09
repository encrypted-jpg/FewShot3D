import torch
import torch.nn as nn
from torch.autograd import Variable

from .pcn import PCNEncoder, PCNEncoderMuVar, PCNGenerator
from .memory_decoder import GPUMemoryModule, MHARawEncoder
from .treegan import (
    Generator,
    Discriminator,
    GradientPenalty,
    TreeGAN,
    DiscriminatorRaw,
    PartGenerator,
)
import pytorch_lightning as pl
from utils.utils import (
    count_parameters,
    AverageMeter,
    plot_image_output_gt,
    reparameterization,
    plot_pcd_one_view,
    CosineAnnealingWarmRestartsDecayLR,
)
from extensions.chamfer_dist import ChamferDistanceL1
from pytorch_metric_learning import distances, losses
from extensions.dcd import DCD
import numpy as np
import os
import wandb
from torch.utils.tensorboard import SummaryWriter


class BicycleGANModule(pl.LightningModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.encoder = None
        self.generator = None
        self.D_VAE = None
        self.D_LR = None
        self.chamfer_loss = ChamferDistanceL1()
        self.dcd_loss = DCD(alpha=alpha)
        self.mae_loss = nn.L1Loss()
        self.gp_loss = GradientPenalty(lambdaGP=lambdaGP)
        self.encoded_priors = {}
        self.Tensor = torch.Tensor

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_D_VAE = torch.optim.Adam(self.D_VAE.parameters(), lr=lr)
        self.optimizer_D_LR = torch.optim.Adam(self.D_LR.parameters(), lr=lr)

    def _setup_schedulers(self, gamma=0.9):
        self.scheduler_E = torch.optim.lr_scheduler.StepLR(
            self.optimizer_E, step_size=1, gamma=gamma
        )
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=1, gamma=gamma
        )
        self.scheduler_D_VAE = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D_VAE, step_size=1, gamma=gamma
        )
        self.scheduler_D_LR = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D_LR, step_size=1, gamma=gamma
        )

    def setup_ops(self):
        self._setup_optimizers(self.args.lr)
        self._setup_schedulers(self.args.gamma)
        self.dir = os.path.join(self.args.log_dir, self.args.exp)
        self.validation_step_loss = AverageMeter()
        self.train_step_loss = AverageMeter()
        self.best_train_step_loss = 1e10
        self.best_val_step_loss = 1e10
        self.best_val_loss_epoch = -1
        self.train_step = 0
        self.writer: SummaryWriter = None

    def to(self, device):
        self.encoder.to(device)
        self.generator.to(device)
        self.D_VAE.to(device)
        self.D_LR.to(device)
        self.chamfer_loss.to(device)
        self.dcd_loss.to(device)
        self.gp_loss.to(device)
        self.mae_loss.to(device)
        if device.type == "cuda":
            self.Tensor = torch.cuda.FloatTensor
        return self

    def on_train_epoch_start(self):
        self.log("lr", self.scheduler_G.get_last_lr()[0], prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch

        valid = 1
        fake = 0

        real_A = Variable(A.type(self.Tensor))
        real_B = Variable(B.type(self.Tensor))
        for _ in range(self.args.gen_freq):
            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()

            encoded_z = self.encoder(real_B)
            fake_B = self.generator(real_A, encoded_z)

            loss_chamfer = self.chamfer_loss(fake_B, real_B)
            loss_chamfer_dcd = self.dcd_loss(fake_B, real_B)
            loss_VAE_GAN = self.D_VAE.compute_loss(fake_B, valid)

            sampled_z = Variable(
                torch.tensor(
                    np.random.normal(0, 1, (real_A.size(0), self.args.latent_dim)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            _fake_B = self.generator(real_A, sampled_z)

            loss_chamfer_sampled = self.chamfer_loss(_fake_B, real_B)
            loss_chamfer_dcd_sampled = self.dcd_loss(_fake_B, real_B)
            _loss_chamfer_dcd_sampled = loss_chamfer_dcd_sampled * 10

            loss_LR_GAN = self.D_LR.compute_loss(_fake_B, valid)

            loss_GE = (
                loss_VAE_GAN + loss_LR_GAN + self.args.lambda_chamfer * loss_chamfer_dcd
            )
            loss_GE *= 10
            loss_GE.backward(retain_graph=True)
            self.optimizer_E.step()

            _encoded_z = self.encoder(_fake_B)
            loss_latent = self.args.lambda_latent * self.mae_loss(_encoded_z, sampled_z)

            loss_latent.backward(retain_graph=True)
            _loss_chamfer_dcd_sampled.backward()
            self.optimizer_G.step()

        for i in range(real_B.shape[0]):
            if taxonomy_id[i] not in self.encoded_priors:
                self.encoded_priors[taxonomy_id[i]] = AverageMeter()
            self.encoded_priors[taxonomy_id[i]].update(
                encoded_z[i].detach().cpu().numpy()
            )

        self.optimizer_D_VAE.zero_grad()

        loss_D_VAE = (
            self.D_VAE.compute_loss(real_B, valid)
            + self.D_VAE.compute_loss(fake_B.detach(), fake)
            + self.gp_loss(self.D_VAE, real_B.data, fake_B.data)
        )

        loss_D_VAE.backward()
        self.optimizer_D_VAE.step()

        self.optimizer_D_LR.zero_grad()

        loss_D_LR = (
            self.D_LR.compute_loss(real_B, valid)
            + self.D_LR.compute_loss(_fake_B.detach(), fake)
            + self.gp_loss(self.D_LR, real_B.data, _fake_B.data)
        )

        loss_D_LR.backward()
        self.optimizer_D_LR.step()

        loss_dict = {
            "loss_GE": loss_GE.item(),
            "loss_latent": loss_latent.item(),
            "loss_D_VAE": loss_D_VAE.item(),
            "loss_D_LR": loss_D_LR.item(),
            "loss_chamfer": loss_chamfer.item() * 1000,
            "loss_chamfer_dcd": loss_chamfer_dcd.item() * 1000,
            "loss_chamfer_sampled": loss_chamfer_sampled.item() * 1000,
            "loss_chamfer_dcd_sampled": loss_chamfer_dcd_sampled.item() * 1000,
            "loss_VAE_GAN": loss_VAE_GAN.item(),
            "loss_LR_GAN": loss_LR_GAN.item(),
        }
        self.log_dict(loss_dict)
        self.log(
            "loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_step_loss.update(loss_chamfer.item() * 1000)
        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, self.train_step)
        self.handle_post_train_step(real_A, real_B, fake_B, _fake_B)
        return loss_dict

    def handle_post_train_step(self, real_A, real_B, fake_B, _fake_B):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, real_A.shape[0])
            plot_image_output_gt(
                os.path.join(self.dir, "train", f"train_{self.train_step}.png"),
                real_A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(),
                fake_B[index].detach().cpu().numpy(),
                real_B[index].detach().cpu().numpy(),
            )

        if self.train_step % 100 == 0:
            index = np.random.randint(0, real_A.shape[0])
            wandb.log(
                {
                    "train_B": wandb.Object3D(real_B[index].detach().cpu().numpy()),
                    "train_B_fake": wandb.Object3D(
                        fake_B[index].detach().cpu().numpy()
                    ),
                    "train_B_fake_Sampled": wandb.Object3D(
                        _fake_B[index].detach().cpu().numpy()
                    ),
                    "train_A": wandb.Image(
                        real_A[index]
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
                    "train_A", real_A[index].detach().cpu().numpy(), self.train_step
                )
                self.writer.add_mesh(
                    "train_B",
                    vertices=real_B[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.add_mesh(
                    "train_fake_B",
                    vertices=fake_B[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.add_mesh(
                    "train_fake_B_Sampled",
                    vertices=_fake_B[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.flush()

    def on_train_epoch_end(self):
        self.scheduler_E.step()
        self.scheduler_G.step()
        self.scheduler_D_VAE.step()
        self.scheduler_D_LR.step()
        self.writer.add_scalar(
            "train_loss_epoch", self.train_step_loss.avg, self.current_epoch
        )
        self.writer.flush()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch
        real_A = Variable(A.type(self.Tensor))
        real_B = Variable(B.type(self.Tensor))

        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()

        prior_z = []
        for i in range(real_A.shape[0]):
            if taxonomy_id[i] in self.encoded_priors:
                prior_z.append(self.encoded_priors[taxonomy_id[i]].avg)
            else:
                prior_z.append(np.random.normal(0, 1, (self.args.latent_dim)))
        prior_z = np.array(prior_z)
        prior_z = Variable(self.Tensor(prior_z))
        _fake_B = self.generator(real_A, prior_z)

        loss_chamfer = self.chamfer_loss(_fake_B, real_B)
        self.log(
            "val_loss_chamfer",
            loss_chamfer.item() * 1000,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.validation_step_loss.update(loss_chamfer.item() * 1000)
        self.writer.add_scalar(
            "val_loss", loss_chamfer.item() * 1000, self.current_epoch
        )
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
            "lr", self.scheduler_G.get_last_lr()[0], self.current_epoch
        )
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            "encoder": self.encoder.state_dict(),
            "generator": self.generator.state_dict(),
            "D_VAE": self.D_VAE.state_dict(),
            "D_LR": self.D_LR.state_dict(),
            "encoded_priors": self.encoded_priors,
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
        self.encoder.load_state_dict(data["encoder"])
        self.generator.load_state_dict(data["generator"])
        self.D_VAE.load_state_dict(data["D_VAE"])
        self.D_LR.load_state_dict(data["D_LR"])
        if "encoded_priors" in data:
            self.encoded_priors = data["encoded_priors"]
        # self.best_val_step_loss = data['val_loss']
        # self.best_train_step_loss = data['train_loss']
        print("Model Loaded Successfully")


class BicycleGANMuVarModule(BicycleGANModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__(args, alpha, lambdaGP)

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (A, B, C) = batch

        valid = 1
        fake = 0

        real_A = Variable(A.type(self.Tensor))
        real_B = Variable(B.type(self.Tensor))
        for _ in range(self.args.gen_freq):
            self.optimizer_E.zero_grad()
            self.optimizer_G.zero_grad()

            mu, logvar = self.encoder(real_B)
            encoded_z = reparameterization(mu, logvar, self.device, self.args)
            fake_B = self.generator(real_A, encoded_z)

            loss_chamfer = self.chamfer_loss(fake_B, real_B)
            loss_chamfer_dcd = self.dcd_loss(fake_B, real_B)
            loss_VAE_GAN = self.D_VAE.compute_loss(fake_B, valid)
            loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar - 1)

            sampled_z = Variable(
                torch.tensor(
                    np.random.normal(0, 1, (real_A.size(0), self.args.latent_dim)),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            _fake_B = self.generator(real_A, sampled_z)

            loss_chamfer_sampled = self.chamfer_loss(_fake_B, real_B)
            loss_chamfer_dcd_sampled = self.dcd_loss(_fake_B, real_B)
            _loss_chamfer_dcd_sampled = loss_chamfer_dcd_sampled * 10

            loss_LR_GAN = self.D_LR.compute_loss(_fake_B, valid)

            loss_GE = (
                loss_VAE_GAN
                + loss_LR_GAN
                + self.args.lambda_chamfer * loss_chamfer_dcd
                + loss_kl * self.args.lambda_kl
            )
            loss_GE *= 10
            loss_GE.backward(retain_graph=True)
            self.optimizer_E.step()

            _mu, _ = self.encoder(_fake_B)
            loss_latent = self.args.lambda_latent * self.mae_loss(_mu, sampled_z)

            loss_latent.backward(retain_graph=True)
            _loss_chamfer_dcd_sampled.backward()
            self.optimizer_G.step()

        for i in range(real_B.shape[0]):
            if taxonomy_id[i] not in self.encoded_priors:
                self.encoded_priors[taxonomy_id[i]] = AverageMeter()
            self.encoded_priors[taxonomy_id[i]].update(mu[i].detach().cpu().numpy())

        self.optimizer_D_VAE.zero_grad()

        loss_D_VAE = (
            self.D_VAE.compute_loss(real_B, valid)
            + self.D_VAE.compute_loss(fake_B.detach(), fake)
            + self.gp_loss(self.D_VAE, real_B.data, fake_B.data)
        )

        loss_D_VAE.backward()
        self.optimizer_D_VAE.step()

        self.optimizer_D_LR.zero_grad()

        loss_D_LR = (
            self.D_LR.compute_loss(real_B, valid)
            + self.D_LR.compute_loss(_fake_B.detach(), fake)
            + self.gp_loss(self.D_LR, real_B.data, _fake_B.data)
        )

        loss_D_LR.backward()
        self.optimizer_D_LR.step()

        loss_dict = {
            "loss_GE": loss_GE.item(),
            "loss_latent": loss_latent.item(),
            "loss_D_VAE": loss_D_VAE.item(),
            "loss_D_LR": loss_D_LR.item(),
            "loss_chamfer": loss_chamfer.item() * 1000,
            "loss_chamfer_dcd": loss_chamfer_dcd.item() * 1000,
            "loss_chamfer_sampled": loss_chamfer_sampled.item() * 1000,
            "loss_chamfer_dcd_sampled": loss_chamfer_dcd_sampled.item() * 1000,
            "loss_VAE_GAN": loss_VAE_GAN.item(),
            "loss_LR_GAN": loss_LR_GAN.item(),
            "loss_kl": loss_kl.item(),
        }
        self.log_dict(loss_dict)
        self.log(
            "loss",
            loss_chamfer,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.train_step_loss.update(loss_chamfer.item() * 1000)
        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, self.train_step)
        self.handle_post_train_step(real_A, real_B, fake_B, _fake_B)
        return loss_dict


class TreeGANModule(BicycleGANModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__(args, alpha, lambdaGP)
        self.encoder = PCNEncoder(latent_dim=args.latent_dim)
        self.generator = Generator(batch_size=args.batch_size)
        self.D_VAE = Discriminator(batch_size=args.batch_size)
        self.D_LR = Discriminator(batch_size=args.batch_size)
        self.setup_ops()


class TreeGANMuVarModule(BicycleGANMuVarModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__(args, alpha, lambdaGP)
        self.encoder = PCNEncoderMuVar(latent_dim=args.latent_dim)
        self.generator = Generator(batch_size=args.batch_size)
        self.D_VAE = Discriminator(batch_size=args.batch_size)
        self.D_LR = Discriminator(batch_size=args.batch_size)
        self.setup_ops()


class PCNGANModule(BicycleGANModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__(args, alpha, lambdaGP)
        self.encoder = PCNEncoder(latent_dim=args.latent_dim)
        self.generator = PCNGenerator(
            latent_dim=args.latent_dim, num_dense=args.num_dense
        )
        self.D_VAE = Discriminator(batch_size=args.batch_size)
        self.D_LR = Discriminator(batch_size=args.batch_size)
        self.encoder_rep_func = self._get_encoder_rep_normal
        self.setup_ops()


class PCNGANMuVarModule(BicycleGANMuVarModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__(args, alpha, lambdaGP)
        self.encoder = PCNEncoderMuVar(latent_dim=args.latent_dim)
        self.generator = PCNGenerator(
            latent_dim=args.latent_dim, num_dense=args.num_dense
        )
        self.D_VAE = Discriminator(batch_size=args.batch_size)
        self.D_LR = Discriminator(batch_size=args.batch_size)
        self.setup_ops()


class TreeGANRawModule(pl.LightningModule):
    def __init__(self, args, alpha=80, lambdaGP=10):
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.generator = TreeGAN(batch_size=args.batch_size)
        self.discriminator = DiscriminatorRaw(batch_size=args.batch_size)
        self.chamfer_loss = ChamferDistanceL1()
        self.dcd_loss = DCD(alpha=alpha)
        self.gp_loss = GradientPenalty(lambdaGP=lambdaGP)
        self.Tensor = torch.Tensor
        # self.setup_ops()

    def _setup_optimizers(self, lr=1e-5):
        self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=lr)
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=lr)

    def _setup_schedulers(self, gamma=0.9, total_steps=1000):
        # self.scheduler_G = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_G, step_size=1, gamma=gamma)
        # self.scheduler_D = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer_D, step_size=1, gamma=gamma)
        self.scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_G, max_lr=4e-5, total_steps=total_steps
        )
        self.scheduler_D = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_D, max_lr=4e-5, total_steps=total_steps
        )

    def setup_ops(self, total_steps=1000):
        self._setup_optimizers(self.args.lr)
        self._setup_schedulers(self.args.gamma, total_steps)
        self.dir = os.path.join(self.args.log_dir, self.args.exp)
        self.validation_step_loss = AverageMeter()
        self.train_step_loss = AverageMeter()
        self.best_train_step_loss = 1e10
        self.best_val_step_loss = 1e10
        self.best_val_loss_epoch = -1
        self.train_step = 0
        self.writer: SummaryWriter = None

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        self.chamfer_loss.to(device)
        self.dcd_loss.to(device)
        self.gp_loss.to(device)
        if device.type == "cuda":
            self.Tensor = torch.cuda.FloatTensor
        return self

    def training_step(self, batch, batch_idx):
        taxonomy_id, model_id, (_, B, _) = batch

        point = B.float()
        for _ in range(self.args.dis_freq):
            self.optimizer_D.zero_grad()
            z = torch.randn(
                self.args.batch_size, 1, self.args.latent_dim * 2, device=self.device
            )  # .to(self.device)
            tree = [z]
            with torch.no_grad():
                fake = self.generator(tree)

            D_real = self.discriminator(point)
            D_realm = D_real.mean()

            D_fake = self.discriminator(fake)
            D_fakem = D_fake.mean()

            gp_loss = self.gp_loss(self.discriminator, point.data, fake.data)

            d_loss = -D_realm + D_fakem
            d_loss_gp = d_loss + gp_loss
            d_loss_gp.backward()
            self.optimizer_D.step()

        self.writer.add_scalar("d_loss", d_loss, self.train_step)

        for _ in range(self.args.gen_freq):
            self.optimizer_G.zero_grad()
            z = torch.randn(
                self.args.batch_size, 1, self.args.latent_dim * 2, device=self.device
            )  # .to(self.device)

            tree = [z]
            fake = self.generator(tree)
            G_fake = self.discriminator(fake)
            G_fakem = G_fake.mean()

            g_loss = -G_fakem
            g_loss.backward()
            self.optimizer_G.step()

        self.writer.add_scalar("g_loss", g_loss, self.train_step)
        loss_dict = {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "lr": self.scheduler_G.get_last_lr()[0],
        }
        self.log_dict(loss_dict)
        self.log(
            "loss", g_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("lr", self.scheduler_G.get_last_lr()[0], prog_bar=True, logger=False)
        self.train_step_loss.update(g_loss)
        for k, v in loss_dict.items():
            self.writer.add_scalar(k, v, self.train_step)
        self.handle_post_train_step(point, fake)
        return loss_dict

    def handle_post_train_step(self, point, fake):
        self.train_step += 1
        if self.train_step % self.args.save_iter == 0:
            index = np.random.randint(0, point.shape[0])
            # plot two images side-by-side, write code using plot_pcd_one_view
            plot_pcd_one_view(
                os.path.join(self.dir, "train", f"train_{self.train_step}.png"),
                [
                    point[index].detach().cpu().numpy(),
                    fake[index].detach().cpu().numpy(),
                ],
                titles=["Real", "Fake"],
            )

        if self.train_step % 500 == 0:
            index = np.random.randint(0, point.shape[0])
            wandb.log(
                {
                    "train_B": wandb.Object3D(point[index].detach().cpu().numpy()),
                    "train_B_fake": wandb.Object3D(fake[index].detach().cpu().numpy()),
                }
            )
            if self.writer:
                self.writer.add_mesh(
                    "train_B",
                    vertices=point[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.add_mesh(
                    "train_fake_B",
                    vertices=fake[index].detach().cpu().numpy().reshape(1, -1, 3),
                    global_step=self.train_step,
                )
                self.writer.flush()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def on_train_epoch_end(self):
        # self.scheduler_G.step()
        # self.scheduler_D.step()
        self.writer.add_scalar(
            "train_loss_epoch", self.train_step_loss.avg, self.current_epoch
        )
        self.writer.flush()

    def validation_step(self, batch, batch_idx):
        taxonomy_id, model_id, (_, B, _) = batch
        point = B.float()
        z = torch.randn(
            self.args.batch_size, 1, self.args.latent_dim * 2, device=self.device
        )
        tree = [z]
        fake = self.generator(tree)
        fake = fake.float()
        loss = self.chamfer_loss(fake, point)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.validation_step_loss.update(loss)
        self.writer.add_scalar("val_loss", loss, self.current_epoch)
        return

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
        self.writer.flush()
        self.train_step_loss.reset()
        self.validation_step_loss.reset()

    def configure_optimizers(self):
        return None

    def save_model(self, path):
        data = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
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
        self.generator.load_state_dict(data["generator"])
        self.discriminator.load_state_dict(data["discriminator"])
        # self.best_val_step_loss = data['val_loss']
        # self.best_train_step_loss = data['train_loss']
        print("Model Loaded Successfully")


# class MHATreeGANModule(pl.LightningModule):
#     def __init__(self, args, alpha=80, lambdaGP=10):
#         super().__init__()
#         self.args = args
#         self.save_hyperparameters()
#         self.automatic_optimization = False
#         self.embed_dim = 1024
#         self.pc_feat_encoder = PCNEncoder(latent_dim=self.embed_dim)
#         self.img_encoder = BaseImageEncoder(latent_dim=self.embed_dim)
#         self.mha_encoder = MHARawEncoder(latent_dim=self.embed_dim, num_heads=4)
#         self.generator = PartGenerator(
#             latent_dim=self.embed_dim, batch_size=args.batch_size
#         )
#         self.D_VAE = Discriminator(batch_size=args.batch_size)
#         self.D_LR = Discriminator(batch_size=args.batch_size)
#         self.memory = GPUMemoryModule(
#             capacity=4000, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
#         )
#         self.val_memory = GPUMemoryModule(
#             capacity=400, img_embed_dim=self.embed_dim, pc_embed_dim=self.embed_dim
#         )
#         self.pc_feat_encoder.requires_grad_(False)
#         self.chamfer_loss = ChamferDistanceL1()
#         self.dcd_loss = DCD(alpha=alpha)
#         self.mae_loss = nn.L1Loss()
#         self.gp_loss = GradientPenalty(lambdaGP=lambdaGP)
#         self.Tensor = torch.Tensor
#         self.ms_loss = losses.MultiSimilarityLoss(
#             distance=distances.LpDistance(normalize_embeddings=False)
#         )
#         self.setup_ops()

#     def _setup_optimizers(self):
#         self.optimizer_G = torch.optim.AdamW(
#             self.generator.parameters(), lr=self.args.lr
#         )
#         self.optimizer_D_VAE = torch.optim.AdamW(
#             self.D_VAE.parameters(), lr=self.args.lr
#         )
#         self.optimizer_D_LR = torch.optim.AdamW(self.D_LR.parameters(), lr=self.args.lr)
#         self.optimizer_ME = torch.optim.AdamW(
#             self.mha_encoder.parameters(), lr=self.args.lr
#         )
#         self.optimizer_BE = torch.optim.AdamW(
#             self.img_encoder.parameters(), lr=self.args.lr
#         )

#     def _setup_schedulers(self):
#         self.scheduler_G = CosineAnnealingWarmRestartsDecayLR(
#             self.optimizer_ME,
#             T_0=self.args.T_0,
#             T_mult=2,
#             eta_min=5e-8,
#             decay_factor=self.args.decay_factor,
#         )
#         self.scheduler_D_VAE = CosineAnnealingWarmRestartsDecayLR(
#             self.optimizer_D_VAE,
#             T_0=self.args.T_0,
#             T_mult=2,
#             eta_min=5e-8,
#             decay_factor=self.args.decay_factor,
#         )
#         self.scheduler_D_LR = CosineAnnealingWarmRestartsDecayLR(
#             self.optimizer_D_LR,
#             T_0=self.args.T_0,
#             T_mult=2,
#             eta_min=5e-8,
#             decay_factor=self.args.decay_factor,
#         )
#         self.scheduler_ME = CosineAnnealingWarmRestartsDecayLR(
#             self.optimizer_ME,
#             T_0=self.args.T_0,
#             T_mult=2,
#             eta_min=5e-8,
#             decay_factor=self.args.decay_factor,
#         )
#         self.scheduler_BE = CosineAnnealingWarmRestartsDecayLR(
#             self.optimizer_BE,
#             T_0=self.args.T_0,
#             T_mult=2,
#             eta_min=5e-8,
#             decay_factor=self.args.decay_factor,
#         )

#     def setup_ops(self):
#         self._setup_optimizers()
#         self._setup_schedulers()
#         self.dir = os.path.join(self.args.log_dir, self.args.exp)
#         self.validation_step_loss = AverageMeter()
#         self.train_step_loss = AverageMeter()
#         self.best_train_step_loss = 1e10
#         self.best_val_step_loss = 1e10
#         self.best_val_loss_epoch = -1
#         self.train_step = 0
#         self.val_step = 0
#         self.writer: SummaryWriter = None
#         self.test_encodings = []
#         self.test_cats = []

#     def on_train_epoch_start(self):
#         self.memory.clear()
#         self.tmem_log = False
#         self.vmem_log = False
#         self.mha_encoder.set_memory(self.memory)

#     def training_step(self, batch, batch_idx):
#         taxonomy_id, model_id, (A, B, C) = batch

#         valid = 1
#         fake = 0

#         real_A = Variable(A.type(self.Tensor))
#         real_B = Variable(B.type(self.Tensor))
#         for _ in range(self.args.gen_freq):
#             self.optimizer_BE.zero_grad(set_to_none=True)
#             self.optimizer_G.zero_grad(set_to_none=True)
#             self.optimizer_ME.zero_grad(set_to_none=True)

#             pc_feat = self.pc_feat_encoder(real_B)
#             img_feat = self.img_encoder(real_A)
#             encoded_z = self.mha_encoder(img_feat)
#             fake_B = self.generator(img_feat, encoded_z)

#             loss_chamfer = self.chamfer_loss(fake_B, real_B)
#             loss_chamfer_dcd = self.dcd_loss(fake_B, real_B)
#             loss_VAE_GAN = self.D_VAE.compute_loss(fake_B, valid)

#             sampled_z = Variable(
#                 torch.tensor(
#                     np.random.normal(0, 1, (real_A.size(0), self.args.latent_dim)),
#                     dtype=torch.float32,
#                     device=self.device,
#                 )
#             )
#             _fake_B = self.generator(img_feat, sampled_z)

#             loss_chamfer_sampled = self.chamfer_loss(_fake_B, real_B)
#             loss_chamfer_dcd_sampled = self.dcd_loss(_fake_B, real_B)
#             _loss_chamfer_dcd_sampled = loss_chamfer_dcd_sampled * 10

#             loss_LR_GAN = self.D_LR.compute_loss(_fake_B, valid)

#             loss_GE = (
#                 loss_VAE_GAN + loss_LR_GAN + self.args.lambda_chamfer * loss_chamfer_dcd
#             )
#             loss_GE *= 10
#             loss_GE.backward()
#             self.optimizer_BE.step()
#             self.optimizer_ME.step()

#             _encoded_z = self.pc_feat_encoder(_fake_B)
#             loss_latent = self.args.lambda_latent * self.mae_loss(_encoded_z, sampled_z)

#             loss_latent.backward(retain_graph=True)
#             _loss_chamfer_dcd_sampled.backward()
#             self.optimizer_G.step()

#         for i in range(real_B.shape[0]):
#             if taxonomy_id[i] not in self.encoded_priors:
#                 self.encoded_priors[taxonomy_id[i]] = AverageMeter()
#             self.encoded_priors[taxonomy_id[i]].update(
#                 encoded_z[i].detach().cpu().numpy()
#             )

#         self.optimizer_D_VAE.zero_grad()

#         # loss_D_VAE = (
