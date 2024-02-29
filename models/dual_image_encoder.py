from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout
import torch
from .base_image_encoder import BaseImageEncoder


class DualImageEncoder(nn.Module):
    def __init__(self, latent_dim=1024, percent_second=0.2):
        super(DualImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.percent_second = percent_second
        self.first_encoder = BaseImageEncoder(latent_dim)
        self.second_encoder = BaseImageEncoder(latent_dim)
        self.dropout1 = Dropout(p=0.3)
        self.dropout2 = Dropout(p=0.4)
        out1 = int(latent_dim * (1 - percent_second))
        self.fc1 = nn.Linear(latent_dim, 2 * out1)
        self.fc2 = nn.Linear(latent_dim, 2 * (latent_dim - out1))
        self.fc3 = nn.Linear(2 * latent_dim, latent_dim)

    def forward(self, img1, img2):
        img1 = self.first_encoder(img1)
        img2 = self.second_encoder(img2)
        img1 = self.dropout1(img1)
        img2 = self.dropout1(img2)
        img1 = F.relu(self.fc1(img1))
        img2 = F.relu(self.fc2(img2))
        x = torch.cat((img1, img2), dim=1)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
