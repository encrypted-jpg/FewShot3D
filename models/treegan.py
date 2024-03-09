import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable, grad

from .base_image_encoder import BaseImageEncoder

# from .memory_decoder import MHAEncoder, MultiHeadAttention


class TreeGCN(nn.Module):
    def __init__(
        self,
        batch,
        depth,
        features,
        degrees,
        support=10,
        node=1,
        upsample=False,
        activation=True,
    ):
        self.batch = batch
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList(
            [
                nn.Linear(features[inx], self.out_feature, bias=False)
                for inx in range(self.depth + 1)
            ]
        )

        if self.upsample:
            self.W_branch = nn.Parameter(
                torch.FloatTensor(
                    self.node, self.in_feature, self.degree * self.in_feature
                )
            )

        self.W_loop = nn.Sequential(
            nn.Linear(self.in_feature, self.in_feature * support, bias=False),
            nn.Linear(self.in_feature * support, self.out_feature, bias=False),
        )

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain("relu"))

        stdv = 1.0 / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        root = 0
        for inx in range(self.depth + 1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1, 1, repeat_num).view(
                self.batch, -1, self.out_feature
            )

        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(self.batch, self.node * self.degree, self.in_feature)

            branch = self.W_loop(branch)

            branch = (
                root.repeat(1, 1, self.degree).view(self.batch, -1, self.out_feature)
                + branch
            )
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1, self.node, 1))
        tree.append(branch)

        return tree


class Discriminator(nn.Module):
    def __init__(self, batch_size, features=[3, 64, 128, 256, 512, 1024]):
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(
                nn.Conv1d(features[inx], features[inx + 1], kernel_size=1, stride=1)
            )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(features[-1], features[-3]),
            nn.Linear(features[-3], features[-5]),
            nn.Linear(features[-5], 2),
        )
        self.out_act = nn.Softmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out)  # (B, 1)
        out = self.out_act(out)
        return out

    def compute_loss(self, x, gt):
        # loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        # create a tensor with gt label for each output, here gt is int, not a tensor
        out = self.forward(x)
        gt = Variable(torch.tensor([gt] * x.size(0))).to(x.device)
        loss = self.cross_entropy(out, gt)
        return loss


class DiscriminatorRaw(nn.Module):
    def __init__(self, batch_size, features=[3, 64, 128, 256, 512, 1024]):
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        super(DiscriminatorRaw, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(
                nn.Conv1d(features[inx], features[inx + 1], kernel_size=1, stride=1)
            )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(features[-1], features[-3]),
            nn.Linear(features[-3], features[-5]),
            nn.Linear(features[-5], 1),
        )

    def forward(self, f):
        feat = f.transpose(1, 2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out)  # (B, 1)

        return out


class TreeGAN(nn.Module):
    def __init__(
        self,
        batch_size=4,
        features=[512, 64, 64, 64, 64, 32, 3],
        degrees=[2, 2, 2, 4, 8, 32],
        support=10,
    ):
        self.batch_size = batch_size
        self.layer_num = len(features) - 1
        assert self.layer_num == len(
            degrees
        ), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(TreeGAN, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        self.batch_size,
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=False,
                    ),
                )
            else:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        self.batch_size,
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=True,
                    ),
                )
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]


class RawGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=2048,
        batch_size=4,
        features=[2048, 256, 64, 64, 64, 32, 3],
        degrees=[2, 2, 2, 4, 8, 32],
        support=10,
    ):
        super(RawGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        assert latent_dim == features[0]
        self.generator = TreeGAN(batch_size, features, degrees, support)

    def forward(self, z):
        z = z.unsqueeze(1)
        tree = [z]
        pointcloud = self.generator(tree)
        return pointcloud


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        batch_size=4,
        features=[512, 64, 64, 64, 64, 32, 3],
        degrees=[2, 2, 2, 4, 8, 32],
        support=10,
    ):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.image_encoder = BaseImageEncoder(latent_dim=latent_dim)
        self.generator = TreeGAN(batch_size, features, degrees, support)

    def forward(self, img, z):
        img = self.image_encoder(img)
        z = torch.cat((img, z), dim=1).unsqueeze(1)
        tree = [z]
        pointcloud = self.generator(tree)
        return pointcloud


class PartGenerator(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        percent=0.8,
        dropout=0.4,
        batch_size=4,
        features=[2048, 256, 64, 64, 64, 32, 3],
        degrees=[2, 2, 2, 4, 8, 32],
        support=10,
    ):
        super(PartGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.generator = TreeGAN(batch_size, features, degrees, support)
        out1 = int(2 * latent_dim * percent)
        out2 = 2 * latent_dim - out1
        self.linear1 = nn.Linear(latent_dim, 2 * out1)
        self.linear2 = nn.Linear(latent_dim, 2 * out2)
        self.linear3 = nn.Linear(4 * latent_dim, 2 * latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, img_feat, z):
        img_feat = self.linear1(img_feat)
        z = self.linear2(z)
        img_feat = img_feat.squeeze(1)
        z = z.squeeze(1)
        z = torch.cat((img_feat, z), dim=1)
        z = self.activation(z)
        z = self.dropout(z)
        z = self.linear3(z).unsqueeze(1)
        tree = [z]
        pointcloud = self.generator(tree)
        return pointcloud


# class MHAGenerator(nn.Module):
#     def __init__(
#         self,
#         latent_dim=1024,
#         batch_size=4,
#         k_nearest=5,
#         features=[2048, 256, 64, 64, 64, 32, 3],
#         degrees=[2, 2, 2, 4, 8, 32],
#         support=10,
#     ):
#         super(MHAGenerator, self).__init__()
#         self.latent_dim = latent_dim
#         self.batch_size = batch_size
#         self.memory = None
#         self.k_nearest = k_nearest
#         self.image_encoder = BaseImageEncoder(latent_dim=latent_dim)
#         self.mha = MultiHeadAttention(latent_dim, num_heads=8)
#         self.generator = PartGenerator(
#             latent_dim=latent_dim,
#             percent=0.8,
#             batch_size=batch_size,
#             features=features,
#             degrees=degrees,
#             support=support,
#         )

#     def forward(self, img, encoded=True):
#         Q = self.image_encoder(img)
#         if encoded:
#             K, V = self.memory.find_k_nearest(Q, self.k_nearest)
#             Q = Q.unsqueeze(1)
#             z = self.mha(Q, K, V)
#             Q = Q.squeeze(1)
#         else:
#             z = Variable(
#                 torch.tensor(
#                     np.random.normal(0, 1, (img.size(0), self.latent_dim)),
#                     dtype=torch.float32,
#                     device=img.device,
#                 )
#             )
#         pointcloud = self.generator(Q, z)
#         return pointcloud, Q

#     def set_memory(self, memory):
#         self.memory = memory


class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device("cpu")):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)

        fake_data = fake_data[:batch_size]

        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).to(self.device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs

        gradients = (
            grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            .contiguous()
            .view(batch_size, -1)
        )

        gradient_penalty = (
            ((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2
        ).mean() * self.lambdaGP

        return gradient_penalty
