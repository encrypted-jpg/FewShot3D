from math import cos, pi
import torch.optim.lr_scheduler as lr_scheduler
import random
from torch.autograd import Variable
import torch
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from PIL import Image
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def prepare_compact_logger(log_dir="log", exp_name="exp"):
    make_dir(log_dir)
    make_dir(os.path.join(log_dir, exp_name))

    logger_path = os.path.join(log_dir, exp_name)
    ckpt_dir = os.path.join(log_dir, exp_name, 'checkpoints')
    train_dir = os.path.join(log_dir, exp_name, 'train')
    test_dir = os.path.join(log_dir, exp_name, 'train')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(train_dir)
    make_dir(test_dir)


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
    # epochs_dir = os.path.join(log_dir, exp_name, 'epochs')
    test_dir = os.path.join(log_dir, exp_name, 'test')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    # make_dir(epochs_dir)
    make_dir(test_dir)

    logger_file = os.path.join(log_dir, exp_name, 'logger.log')
    log_fd = open(logger_file, 'a')

    test_logger_file = os.path.join(log_dir, exp_name, 'test_logger.log')
    test_log_fd = open(test_logger_file, 'a')

    print_log(log_fd, "Experiment: {}".format(exp_name), False)
    print_log(log_fd, "Logger directory: {}".format(logger_path), False)

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, log_fd, train_writer, val_writer, logger_path, test_dir, test_log_fd


def save_imgs(img1, img2, img3, img4, path):
    stripe_width = 10  # Adjust as needed
    image_height = 224  # Assuming all images have the same height

    # Create a new black stripe image
    black_stripe = Image.new("RGB", (stripe_width, image_height), (0, 0, 0))

    # Concatenate the images horizontally with the black stripe in between
    concatenated_image = Image.new("RGB", (4 * 224 + 3 * stripe_width, 224))
    concatenated_image.paste(img1, (0, 0))
    concatenated_image.paste(black_stripe, (224, 0))
    concatenated_image.paste(img2, (224 + stripe_width, 0))
    concatenated_image.paste(black_stripe, (2 * 224 + stripe_width, 0))
    concatenated_image.paste(img3, (2 * 224 + 2 * stripe_width, 0))
    concatenated_image.paste(black_stripe, (3 * 224 + 2 * stripe_width, 0))
    concatenated_image.paste(img4, (3 * 224 + 3 * stripe_width, 0))

    # Save the concatenated image
    concatenated_image.save(path)


def save_batch_imgs(real_A, fake_B, path):
    img_samples = None
    fake_B = [x for x in fake_B.data.cpu()]
    mids = [len(fake_B) // 4, len(fake_B) // 2, 3 * len(fake_B) // 4]
    i = 0
    img_list = []
    for img_A, f_B in zip(real_A, fake_B):
        f_B = f_B.view(1, *f_B.shape)
        img_A = img_A.view(1, *img_A.shape)
        img_sample = torch.cat((img_A, f_B), -1)
        img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat(
            (img_samples, img_sample), -2)
        i += 1
        if i in mids:
            img_list.append(img_samples)
            img_samples = None
    img_list.append(img_samples)
    img_samples = torch.cat(img_list, -1)
    save_image(img_samples[0][0], path, normalize=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def plot_image_output_gt(filename, image, output_pcd, gt_pcd, img_title='Image', output_title='Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    fig = plt.figure(figsize=(3*4, 5))
    elev = 30
    azim = -45

    # Plot the image
    ax_img = fig.add_subplot(1, 3, 1)
    ax_img.imshow(image)
    ax_img.set_title(img_title)
    ax_img.set_axis_off()

    # Plot the output point cloud
    color_output = output_pcd[:, 0]
    ax_output = fig.add_subplot(1, 3, 2, projection='3d')
    ax_output.view_init(elev, azim)
    ax_output.scatter(output_pcd[:, 0], output_pcd[:, 1], output_pcd[:, 2], zdir=zdir,
                      c=color_output, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_output.set_title(output_title)
    ax_output.set_axis_off()
    ax_output.set_xlim(xlim)
    ax_output.set_ylim(ylim)
    ax_output.set_zlim(zlim)

    # Plot the ground truth point cloud
    color_gt = gt_pcd[:, 0]
    ax_gt = fig.add_subplot(1, 3, 3, projection='3d')
    ax_gt.view_init(elev, azim)
    ax_gt.scatter(gt_pcd[:, 0], gt_pcd[:, 1], gt_pcd[:, 2], zdir=zdir,
                  c=color_gt, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_gt.set_title(gt_title)
    ax_gt.set_axis_off()
    ax_gt.set_xlim(xlim)
    ax_gt.set_ylim(ylim)
    ax_gt.set_zlim(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def plot_batch_2_image_output_gt(filename, image1, image2, output_pcd, gt_pcd, img1_title='Image', img2_title='Depth', output_title='Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y',
                                 xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    fig = plt.figure(figsize=(4*4, len(image1)*4))
    elev = 30
    azim = -45
    image1 = image1.transpose(0, 2, 3, 1)
    image2 = image2.transpose(0, 2, 3, 1)

    for i in range(len(image1)):
        # Plot the first image
        ax_img = fig.add_subplot(len(image1), 4, i*4 + 1)
        ax_img.imshow(image1[i])
        ax_img.set_title(img1_title)
        ax_img.set_axis_off()

        # Plot the second image
        ax_img2 = fig.add_subplot(len(image1), 4, i*4 + 2)
        ax_img2.imshow(image2[i])
        ax_img2.set_title(img2_title)
        ax_img2.set_axis_off()

        # Plot the output point cloud
        color_output = output_pcd[i][:, 0]
        ax_output = fig.add_subplot(len(image1), 4, i*4 + 3, projection='3d')
        ax_output.view_init(elev, azim)
        ax_output.scatter(output_pcd[i][:, 0], output_pcd[i][:, 1], output_pcd[i][:, 2], zdir=zdir,
                          c=color_output, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax_output.set_title(output_title)
        ax_output.set_axis_off()
        ax_output.set_xlim(xlim)
        ax_output.set_ylim(ylim)
        ax_output.set_zlim(zlim)

        # Plot the ground truth point cloud
        color_gt = gt_pcd[i][:, 0]
        ax_gt = fig.add_subplot(len(image1), 4, i*4 + 4, projection='3d')
        ax_gt.view_init(elev, azim)
        ax_gt.scatter(gt_pcd[i][:, 0], gt_pcd[i][:, 1], gt_pcd[i][:, 2], zdir=zdir,
                      c=color_gt, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax_gt.set_title(gt_title)
        ax_gt.set_axis_off()
        ax_gt.set_xlim(xlim)
        ax_gt.set_ylim(ylim)
        ax_gt.set_zlim(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def plot_2_image_output_gt(filename, image1, image2, output_pcd, gt_pcd, img1_title='Image', img2_title='Depth', output_title='Output PCD', gt_title='Ground Truth PCD', suptitle='', pcd_size=0.5, cmap='Reds', zdir='y',
                           xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    fig = plt.figure(figsize=(4*4, 5))
    elev = 30
    azim = -45

    # Plot the image
    ax_img = fig.add_subplot(1, 4, 1)
    ax_img.imshow(image1)
    ax_img.set_title(img1_title)
    ax_img.set_axis_off()
    ax_img = fig.add_subplot(1, 4, 2)
    ax_img.imshow(image2)
    ax_img.set_title(img2_title)
    ax_img.set_axis_off()

    # Plot the output point cloud
    color_output = output_pcd[:, 0]
    ax_output = fig.add_subplot(1, 4, 3, projection='3d')
    ax_output.view_init(elev, azim)
    ax_output.scatter(output_pcd[:, 0], output_pcd[:, 1], output_pcd[:, 2], zdir=zdir,
                      c=color_output, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_output.set_title(output_title)
    ax_output.set_axis_off()
    ax_output.set_xlim(xlim)
    ax_output.set_ylim(ylim)
    ax_output.set_zlim(zlim)

    # Plot the ground truth point cloud
    color_gt = gt_pcd[:, 0]
    ax_gt = fig.add_subplot(1, 4, 4, projection='3d')
    ax_gt.view_init(elev, azim)
    ax_gt.scatter(gt_pcd[:, 0], gt_pcd[:, 1], gt_pcd[:, 2], zdir=zdir,
                  c=color_gt, s=pcd_size, cmap=cmap, vmin=-1.0, vmax=0.5)
    ax_gt.set_title(gt_title)
    ax_gt.set_axis_off()
    ax_gt.set_xlim(xlim)
    ax_gt.set_ylim(ylim)
    ax_gt.set_zlim(zlim)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                      xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30
    azim = -45
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        color = pcd[:, 0]
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir,
                   c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
                        top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (
            max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) >
                0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reparameterization(mu, logvar, device, args):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.tensor(
        np.random.normal(0, 1, (mu.size(0), args.latent_dim)), dtype=torch.float32, device=device))
    z = sampled_z * std + mu
    return z


class CosineAnnealingWarmRestartsDecayLR(lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm restarts, with a decay factor applied 
    to the initial LR after each restart.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first cycle.
        T_mult (int, optional): A factor to increase the cycle length after a restart.
                                Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        decay_factor (float, optional): Factor by which to decay the initial LR
                                        after each restart. Default: 0.8
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.8, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.decay_time = 0
        self.base_lrs = [optimizer.param_groups[0]['lr']]  # Wrap in a list
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.T_i:
            self.T_i += self.T_i * self.T_mult
            self.decay_time += 1
            self.last_epoch = 0

        return [
            (self.eta_min + (base_lr - self.eta_min) * (1 + cos(pi *
             self.last_epoch / self.T_i)) / 2) * (self.decay_factor ** self.decay_time)
            for base_lr in self.base_lrs
        ]

    def step(self):
        """Conduct one step of learning rate update"""
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
