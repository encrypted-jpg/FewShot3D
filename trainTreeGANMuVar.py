from extensions.dcd import DCD
from extensions.chamfer_dist import ChamferDistanceL1
from utils.utils import LambdaLR, plot_image_output_gt, AverageMeter
from datasets import PCNImageDataset, PCNImageHDF5Dataset
from tensorboardX import SummaryWriter
import random
import datetime
import itertools
import cv2
from PIL import Image
import time
from tqdm import tqdm
from models.treegan import Generator, Discriminator, GradientPenalty
from models.pcn import PCNEncoderMuVar
import json
import argparse
import pandas as pd
import open3d as o3d
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def print_log(fd,  message, time=True):
    if time:
        message = ' ==> '.join(
            [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(log_dir="log", exp_name="exp"):
    # prepare logger directory
    make_dir(log_dir)
    make_dir(os.path.join(log_dir, exp_name))

    logger_path = os.path.join(log_dir, exp_name)
    ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
    epochs_dir = os.path.join(log_dir, exp_name, 'epochs')
    test_dir = os.path.join(log_dir, exp_name, 'test')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)
    make_dir(test_dir)

    logger_file = os.path.join(log_dir, exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    test_logger_file = os.path.join(log_dir, exp_name, 'test_logger.log')
    test_log_fd = open(test_logger_file, 'a')

    print_log(log_fd, "Experiment: {}".format(exp_name), False)
    print_log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, logger_path, test_dir, test_log_fd


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dataLoaders(args):
    print("[+] Loading the data...")
    folder = args.folder
    json = args.json
    batch_size = args.batch_size

    # trainDataset = PCNImageDataset(
    #     folder, json, mode='train', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)
    # testDataset = PCNImageDataset(
    #     folder, json, mode='test', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)
    # valDataset = PCNImageDataset(
    #     folder, json, mode='val', b_tag=args.b_tag, img_height=args.size, img_width=args.size, img_count=args.img_count)

    trainDataset = PCNImageHDF5Dataset(folder, json, mode='train', b_tag=args.b_tag,
                                       img_height=args.size, img_width=args.size, img_count=args.img_count)
    testDataset = PCNImageHDF5Dataset(folder, json, mode='test', b_tag=args.b_tag,
                                      img_height=args.size, img_width=args.size, img_count=args.img_count)
    valDataset = PCNImageHDF5Dataset(folder, json, mode='val', b_tag=args.b_tag,
                                     img_height=args.size, img_width=args.size, img_count=args.img_count)

    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True,
                             drop_last=True, num_workers=8, prefetch_factor=8, persistent_workers=True)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=8, prefetch_factor=8, persistent_workers=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False,
                           drop_last=True, num_workers=8, prefetch_factor=8, persistent_workers=True)
    return trainLoader, testLoader, valLoader


def reparameterization(mu, logvar, device, args):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.tensor(
        np.random.normal(0, 1, (mu.size(0), args.latent_dim)), dtype=torch.float32, device=device))
    z = sampled_z * std + mu
    return z


def get_model(args):
    generator = Generator(batch_size=args.batch_size)
    encoder = PCNEncoderMuVar(args.latent_dim)
    D_VAE = Discriminator(batch_size=args.batch_size)
    D_LR = Discriminator(batch_size=args.batch_size)
    print(
        "[+] Number of parameters in Generator: {:.3f} M".format(count_parameters(generator) / 1e6))
    print(
        "[+] Number of parameters in Encoder: {:.3f} M".format(count_parameters(encoder) / 1e6))
    print(
        "[+] Number of parameters in D_VAE: {:.3f} M".format(count_parameters(D_VAE) / 1e6))
    print(
        "[+] Number of parameters in D_LR: {:.3f} M".format(count_parameters(D_LR) / 1e6))
    return generator, encoder, D_VAE, D_LR


def get_scheduler(optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR, args):
    if args.scheduler == 'step':
        lr_scheduler_G = torch.optim.lr_scheduler.StepLR(
            optimizer_G, step_size=1, gamma=args.gamma)
        lr_scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, step_size=1, gamma=args.gamma)
        lr_scheduler_D_VAE = torch.optim.lr_scheduler.StepLR(
            optimizer_D_VAE, step_size=1, gamma=args.gamma)
        lr_scheduler_D_LR = torch.optim.lr_scheduler.StepLR(
            optimizer_D_LR, step_size=1, gamma=args.gamma)

    else:
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        lr_scheduler_E = torch.optim.lr_scheduler.LambdaLR(
            optimizer_E, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        lr_scheduler_D_VAE = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_VAE, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
        lr_scheduler_D_LR = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_LR, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step)
    return lr_scheduler_G, lr_scheduler_E, lr_scheduler_D_VAE, lr_scheduler_D_LR


def train(models, trainLoader, valLoader, args):
    print("[+] Training the model...")
    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer, exp_path, _, _ = prepare_logger(
        args.log_dir, args.exp)
    bestSavePath = os.path.join(ckpt_dir, "bestModel.pth")
    lastSavePath = os.path.join(ckpt_dir, "lastModel.pth")
    print_log(log_fd, str(args))
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, encoder, D_VAE, D_LR = models
    generator = generator.to(device)
    encoder = encoder.to(device)
    D_VAE = D_VAE.to(device)
    D_LR = D_LR.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)
    dcd = DCD(alpha=100).to(device)
    mae_loss = torch.nn.L1Loss()
    mae_loss = mae_loss.to(device)
    gp_loss = GradientPenalty(lambdaGP=10, device=device)

    # Encoded Priors
    encoded_priors = {}

    # Optimizers & LR schedulers
    optimizer_E = torch.optim.Adam(
        encoder.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_D_VAE = torch.optim.Adam(
        D_VAE.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer_D_LR = torch.optim.Adam(
        D_LR.parameters(), lr=args.lr, weight_decay=1e-5)

    lr_scheduler_G, lr_scheduler_E, lr_scheduler_D_VAE, lr_scheduler_D_LR = get_scheduler(
        optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR, args)

    if args.resume:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        generator.load_state_dict(checkpoint['generator'])
        encoder.load_state_dict(checkpoint['encoder'])
        D_VAE.load_state_dict(checkpoint['D_VAE'])
        D_LR.load_state_dict(checkpoint['D_LR'])
        if 'encoded_priors' in checkpoint:
            encoded_priors = checkpoint['encoded_priors']
        # optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        # optimizer_E.load_state_dict(checkpoint['optimizer_E'])
        # optimizer_D_VAE.load_state_dict(checkpoint['optimizer_D_VAE'])
        # optimizer_D_LR.load_state_dict(checkpoint['optimizer_D_LR'])
        # lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
        # lr_scheduler_E.load_state_dict(checkpoint['lr_scheduler_E'])
        # lr_scheduler_D_VAE.load_state_dict(checkpoint['lr_scheduler_D_VAE'])
        # lr_scheduler_D_LR.load_state_dict(checkpoint['lr_scheduler_D_LR'])
        args.epoch = checkpoint['epoch'] + 1
        # minLoss = checkpoint['loss']
        # minLossEpoch = args.epoch
        lr_scheduler_G, lr_scheduler_E, lr_scheduler_D_VAE, lr_scheduler_D_LR = get_scheduler(
            optimizer_G, optimizer_E, optimizer_D_VAE, optimizer_D_LR, args)
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor
    # Adversarial loss
    valid = 1
    fake = 0

    train_step = 0
    minLoss = 1e10
    minLossEpoch = 0
    ###### Training ######
    for epoch in range(args.epoch, args.n_epochs):
        generator.train()
        encoder.train()
        D_VAE.train()
        D_LR.train()
        print_log(
            log_fd, "------------------Epoch: {}------------------".format(epoch))
        train_loss = 0.0
        for key, _ in encoded_priors.items():
            encoded_priors[key] = AverageMeter()
        loader = tqdm(trainLoader)
        for i, batch in enumerate(loader):
            loader.set_description(f"Loss: {(train_loss/(i+1)):.4f}")
            # Set model input
            taxonomy_id, model_id, (A, B, C) = batch
            # Set model input
            real_A = Variable(A.type(Tensor))
            real_B = Variable(B.type(Tensor))

            # -------------------------------
            #  Train Generator and Encoder
            # -------------------------------

            # ----------
            # cVAE-GAN
            # ----------

            # Produce output using encoding of B (cVAE-GAN)
            for _ in range(2):
                optimizer_E.zero_grad()
                optimizer_G.zero_grad()

                mu, logvar = encoder(real_B)
                encoded_z = reparameterization(mu, logvar, device, args)
                fake_B = generator(real_A, encoded_z)

                loss_chamfer = chamfer(fake_B, real_B)
                loss_chamfer_dcd = dcd(fake_B, real_B)

                # Kullback-Leibler divergence of encoded B
                loss_kl = 0.5 * \
                    torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
                # Adversarial loss
                loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

                # ---------
                # cLR-GAN
                # ---------

                # Produce output using sampled z (cLR-GAN)
                sampled_z = Variable(torch.tensor(np.random.normal(
                    0, 1, (real_A.size(0), args.latent_dim)), dtype=torch.float32, device=device))
                _fake_B = generator(real_A, sampled_z)
                # cLR Loss: Adversarial loss
                loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

                # ----------------------------------
                # Total Loss (Generator + Encoder)
                # ----------------------------------

                # loss_GE = loss_VAE_GAN + loss_LR_GAN + args.lambda_pixel * \
                #     loss_pixel + args.lambda_kl * loss_kl

                loss_GE = loss_VAE_GAN + loss_LR_GAN + args.lambda_chamfer * \
                    loss_chamfer_dcd + args.lambda_kl * loss_kl
                loss_GE *= 10
                loss_GE.backward(retain_graph=True)
                optimizer_E.step()

                # ---------------------
                # Generator Only Loss
                # ---------------------

                # Latent L1 loss
                _mu, _ = encoder(_fake_B)
                loss_latent = args.lambda_latent * mae_loss(_mu, sampled_z)

                loss_latent.backward()
                optimizer_G.step()

            for i in range(real_A.shape[0]):
                if taxonomy_id[i] not in encoded_priors:
                    encoded_priors[taxonomy_id[i]] = AverageMeter()
                encoded_priors[taxonomy_id[i]].update(
                    mu[i].detach().cpu().numpy())

            # ----------------------------------
            #  Train Discriminator (cVAE-GAN)
            # ----------------------------------

            optimizer_D_VAE.zero_grad()

            loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(
                fake_B.detach(), fake) + gp_loss(D_VAE, real_B.data, fake_B.data)

            loss_D_VAE.backward()
            optimizer_D_VAE.step()

            # ---------------------------------
            #  Train Discriminator (cLR-GAN)
            # ---------------------------------

            optimizer_D_LR.zero_grad()

            loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(
                _fake_B.detach(), fake) + gp_loss(D_LR, real_B.data, _fake_B.data)

            loss_D_LR.backward()
            optimizer_D_LR.step()

            train_loss += loss_chamfer.item() * 1000
            train_step += 1

            train_writer.add_scalar('loss_GE', loss_GE.item(), train_step)
            train_writer.add_scalar(
                'loss_D_VAE', loss_D_VAE.item(), train_step)
            train_writer.add_scalar('loss_D_LR', loss_D_LR.item(), train_step)
            train_writer.add_scalar(
                'loss_latent', loss_latent.item(), train_step)
            train_writer.add_scalar(
                'loss_chamfer', loss_chamfer.item() * 1000, train_step)
            train_writer.add_scalar(
                'loss_chamfer_dcd', loss_chamfer_dcd.item() * 1000, train_step)
            train_writer.add_scalar('loss_kl', loss_kl.item(), train_step)

            if train_step % 500 == 0:
                index = random.randint(0, A.shape[0] - 1)
                train_writer.add_image(
                    'train_A', real_A[index].detach().cpu().numpy(), train_step)
                train_writer.add_mesh('train_B', vertices=real_B[index].detach(
                ).cpu().numpy().reshape(1, -1, 3), global_step=train_step)
                train_writer.add_mesh('train_fake_B', vertices=fake_B[index].detach(
                ).cpu().numpy().reshape(1, -1, 3), global_step=train_step)
                train_writer.add_mesh('train_fake_B_Sampled', vertices=_fake_B[index].detach(
                ).cpu().numpy().reshape(1, -1, 3), global_step=train_step)

            if train_step % args.save_iter == 0:
                index = random.randint(0, A.shape[0] - 1)
                plot_image_output_gt(os.path.join(exp_path, f"train_{train_step}.png"), real_A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(),
                                     fake_B[index].detach().cpu().numpy(), real_B[index].detach().cpu().numpy())

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_E.step()
        lr_scheduler_D_VAE.step()
        lr_scheduler_D_LR.step()

        train_loss /= len(trainLoader)
        print_log(log_fd, f"Epoch {epoch} Train Loss: {train_loss}")

        ###### Validation ######
        generator.eval()
        encoder.eval()
        D_VAE.eval()
        D_LR.eval()
        val_loss = 0.0
        with torch.no_grad():
            loader = tqdm(valLoader)
            for i, batch in enumerate(loader):
                loader.set_description(f"Loss: {(val_loss/(i+1)):.4f}")
                # Set model input
                taxonomy_id, model_id, (A, B, C) = batch
                real_A = Variable(A.type(Tensor))
                real_B = Variable(B.type(Tensor))

                # -------------------------------
                #  Train Generator and Encoder
                # -------------------------------

                optimizer_E.zero_grad()
                optimizer_G.zero_grad()

                # ----------
                # cVAE-GAN
                # ----------

                # Produce output using encoding of B (cVAE-GAN)
                # encoded_z = encoder(real_B)
                # encoded_z = reparameterization(mu, logvar, device, args)
                # fake_B = generator(real_A, encoded_z)

                # loss_chamfer = chamfer(fake_B, real_B)
                # Kullback-Leibler divergence of encoded B
                # loss_kl = 0.5 * \
                #     torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
                # Adversarial loss
                # loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

                # ---------
                # cLR-GAN
                # ---------

                # Produce output using sampled z (cLR-GAN)
                # sampled_z = Variable(Tensor(np.random.normal(
                #     0, 1, (real_A.size(0), args.latent_dim))))
                # _fake_B = generator(real_A, sampled_z)

                prior_z = []
                for i in range(real_A.shape[0]):
                    if taxonomy_id[i] in encoded_priors:
                        prior_z.append(encoded_priors[taxonomy_id[i]].avg)
                    else:
                        prior_z.append(np.random.normal(
                            0, 1, (args.latent_dim)))
                prior_z = Variable(Tensor(prior_z))
                _fake_B = generator(real_A, prior_z)

                loss_chamfer = chamfer(_fake_B, real_B)
                # cLR Loss: Adversarial loss
                # loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)

                # ----------------------------------
                # Total Loss (Generator + Encoder)
                # ----------------------------------

                # loss_GE = loss_VAE_GAN + loss_LR_GAN + loss_chamfer * args.lambda_chamfer
                # val_writer.add_scalar('loss_GE', loss_GE.item(), i)
                val_writer.add_scalar(
                    'loss_chamfer', loss_chamfer.item() * 1000, i)
                val_loss += loss_chamfer.item() * 1000

                if i % 200 == 0:
                    index = random.randint(0, A.shape[0] - 1)
                    val_writer.add_image(
                        'val_A', real_A[index].detach().cpu().numpy(), i*(epoch+1))
                    val_writer.add_mesh('val_B', vertices=real_B[index].detach(
                    ).cpu().numpy().reshape(1, -1, 3), global_step=i*(epoch+1))
                    val_writer.add_mesh('val_fake_B', vertices=_fake_B[index].detach(
                    ).cpu().numpy().reshape(1, -1, 3), global_step=i*(epoch+1))

        val_loss /= len(valLoader)
        print_log(
            log_fd, f"Epoch {epoch} Val Loss: {val_loss} Learning Rate: {lr_scheduler_G.get_last_lr()[0]}")

        if val_loss < minLoss:
            minLoss = val_loss
            minLossEpoch = epoch
            torch.save({
                'epoch': epoch,
                'loss': val_loss,
                'generator': generator.state_dict(),
                'encoder': encoder.state_dict(),
                'D_VAE': D_VAE.state_dict(),
                'D_LR': D_LR.state_dict(),
                'encoded_priors': encoded_priors,
                # 'optimizer_G': optimizer_G.state_dict(),
                # 'optimizer_E': optimizer_E.state_dict(),
                # 'optimizer_D_VAE': optimizer_D_VAE.state_dict(),
                # 'optimizer_D_LR': optimizer_D_LR.state_dict(),
                # 'lr_scheduler_G': lr_scheduler_G.state_dict(),
                # 'lr_scheduler_E': lr_scheduler_E.state_dict(),
                # 'lr_scheduler_D_VAE': lr_scheduler_D_VAE.state_dict(),
                # 'lr_scheduler_D_LR': lr_scheduler_D_LR.state_dict(),
            }, bestSavePath)
            print_log(log_fd, f"Epoch {epoch} Best Model Saved")

        torch.save({
            'epoch': epoch,
            'loss': val_loss,
            'generator': generator.state_dict(),
            'encoder': encoder.state_dict(),
            'D_VAE': D_VAE.state_dict(),
            'D_LR': D_LR.state_dict(),
            'encoded_priors': encoded_priors,
            # 'optimizer_G': optimizer_G.state_dict(),
            # 'optimizer_E': optimizer_E.state_dict(),
            # 'optimizer_D_VAE': optimizer_D_VAE.state_dict(),
            # 'optimizer_D_LR': optimizer_D_LR.state_dict(),
            # 'lr_scheduler_G': lr_scheduler_G.state_dict(),
            # 'lr_scheduler_E': lr_scheduler_E.state_dict(),
            # 'lr_scheduler_D_VAE': lr_scheduler_D_VAE.state_dict(),
            # 'lr_scheduler_D_LR': lr_scheduler_D_LR.state_dict(),
        }, lastSavePath)

        print_log(log_fd, "Last Model saved (best loss {:.4f} at epoch {})" .format(
            minLoss, minLossEpoch))


def test(models, testLoader, args):
    _, _, _, _, _, _, exp_path, log_fd = prepare_logger(
        args.log_dir, args.exp)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    generator, encoder, D_VAE, D_LR = models
    generator = generator.to(device)
    # encoder = encoder.to(device)
    # D_VAE = D_VAE.to(device)
    # D_LR = D_LR.to(device)

    # Lossess
    chamfer = ChamferDistanceL1().to(device)
    mae_loss = torch.nn.L1Loss()
    mae_loss = mae_loss.to(device)

    # Optimizers & LR schedulers
    # optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    # optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=args.lr)
    # optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=args.lr)

    if args.modelPath:
        print_log(log_fd, f"Loading checkpoint from {args.modelPath}")
        checkpoint = torch.load(args.modelPath)
        generator.load_state_dict(checkpoint['generator'])
        # encoder.load_state_dict(checkpoint['encoder'])
        # D_VAE.load_state_dict(checkpoint['D_VAE'])
        # D_LR.load_state_dict(checkpoint['D_LR'])
        args.epoch = checkpoint['epoch'] + 1
        print_log(
            log_fd, f"Checkpoint loaded (epoch {checkpoint['epoch']}, loss {checkpoint['loss']})")

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor
    valid = 1
    fake = 0
    test_loss = 0.0

    generator.eval()
    # encoder.eval()
    # D_VAE.eval()
    # D_LR.eval()
    key_loss = {}
    with torch.no_grad():
        loader = tqdm(testLoader)
        for i, batch in enumerate(loader):
            loader.set_description(f"Loss: {(test_loss/(i+1)):.4f}")
            # Set model input
            taxonomy_id, model_id, (A, B, C) = batch
            real_A = Variable(A.type(Tensor))
            real_B = Variable(B.type(Tensor))

            # -------------------------------
            #  Train Generator and Encoder
            # -------------------------------

            # optimizer_E.zero_grad()
            optimizer_G.zero_grad()

            # ----------
            # cVAE-GAN
            # ----------

            # Produce output using encoding of B (cVAE-GAN)
            # encoded_z = encoder(real_B)
            # encoded_z = reparameterization(mu, logvar, device, args)
            # fake_B = generator(real_A, encoded_z)

            # loss_chamfer = chamfer(fake_B, real_B)
            # Kullback-Leibler divergence of encoded B
            # loss_kl = 0.5 * \
            #     torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
            # Adversarial loss
            # loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)

            # ---------
            # cLR-GAN
            # ---------

            # Produce output using sampled z (cLR-GAN)
            sampled_z = Variable(Tensor(np.random.normal(
                0, 1, (real_A.size(0), args.latent_dim))))
            fake_B = generator(real_A, sampled_z)

            loss_chamfer = chamfer(fake_B, real_B)
            test_loss += loss_chamfer.item() * 1000

            # for i in range(real_A.shape[0]):
            #     curr_loss = chamfer(fake_B[i].unsqueeze(
            #         0), real_B[i].unsqueeze(0)) * 1000
            #     if taxonomy_id[i] not in key_loss:
            #         key_loss[taxonomy_id[i]] = [curr_loss.item()]
            #     else:
            #         key_loss[taxonomy_id[i]].append((curr_loss.item()))
            if args.testSave:
                for index in range(real_A.shape[0]):
                    plot_image_output_gt(os.path.join(exp_path, f"test_{i}_{index}.png"), real_A[index].detach().cpu().transpose(1, 0).transpose(1, 2).numpy(),
                                         fake_B[index].detach().cpu().numpy(), real_B[index].detach().cpu().numpy())

    test_loss /= len(valLoader)
    print_log(log_fd, f"Test Loss: {test_loss}")
    # print_log(log_fd, "Taxonomy Losses")
    # for key, value in key_loss.items():
    #     print_log(log_fd, f"{key}\t{sum(value)/len(value)}")
    # # save dictionary as pandas dataframe
    # df = pd.DataFrame.from_dict({key: round(sum(value)/len(value), 4)
    #                             for key, value in key_loss.items()}, orient='index')
    # # sort rows based on first column
    # df = df.sort_values(by=[0], ascending=False)
    # df.to_csv(os.path.join(exp_path, 'test.csv'))
    # for key, value in key_loss.items():
    #     print(round(sum(value)/len(value), 3))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default="../ShapeNet", help="Folder containing the data")
    parser.add_argument("--json", type=str, default="final.json",
                        help="JSON file containing the data")
    parser.add_argument("--b_tag", type=str, default="depth",
                        help="Tag for the B Image")
    parser.add_argument("--img_count", type=int, default=3, help="Image count")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log dir")
    parser.add_argument("--exp", type=str, default="exp", help="Experiment")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--size", type=int, default=256, help="Image size")
    parser.add_argument("--latent_dim", type=int,
                        default=256, help="Latent dimension")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch to start")
    parser.add_argument("--scheduler", type=str,
                        default="step", help="Scheduler")
    parser.add_argument("--gamma", type=float, default=0.85, help="Gamma")
    parser.add_argument("--n_epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--decay_epoch", type=int,
                        default=10, help="Decay epoch")
    parser.add_argument("--save_iter", type=int,
                        default=20, help="Save interval")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("--modelPath", type=str,
                        default="bestModel.pth", help="Path to model")
    parser.add_argument("--test", action="store_true", help="Test model")
    parser.add_argument("--testSave", action="store_true",
                        help="Save test output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training")
    parser.add_argument("--lambda_pixel", type=float,
                        default=10, help="pixelwise loss weight")
    parser.add_argument("--lambda_latent", type=float,
                        default=0.5, help="latent loss weight")
    parser.add_argument("--lambda_kl", type=float,
                        default=0.01, help="kullback-leibler loss weight")
    parser.add_argument("--lambda_chamfer", type=float,
                        default=1, help="chamfer loss weight")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    trainLoader, testLoader, valLoader = dataLoaders(args)
    models = get_model(args)
    if args.test:
        test(models, testLoader, args)
    else:
        train(models, trainLoader, valLoader, args)
