from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models


class BaseImageEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super(BaseImageEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.model = models.efficientnet_b4()
        self.load_efficient_net()
        # unfreeze the last layer only
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.model.classifier.parameters():
        #     param.requires_grad = True
        dim = max(latent_dim * 2, 1024)
        self.fc1 = nn.Linear(1000, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, latent_dim)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.GELU()

    def load_efficient_net(self):
        url = (
            "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"
        )
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
