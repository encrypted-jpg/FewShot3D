from numpy import require
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSNNLoss(nn.Module):
    def __init__(self, temperature=50):
        super(BaseSNNLoss, self).__init__()
        self.temperature = temperature

    def forward(self, reps, labels):
        reps = F.normalize(reps, dim=1)

        positive_pairs = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_pairs.fill_diagonal_(False)  # Remove self-pairs

        similarities = torch.matmul(reps, reps.T) / self.temperature
        positive_similarities = similarities[positive_pairs]

        pexp = torch.mean(torch.exp(positive_similarities))
        texp = torch.sum(torch.exp(similarities))

        out = -torch.log(pexp / texp)
        return out


class CDSNNL(nn.Module):
    def __init__(self, chamfer_dist_sep, delta=0.01, gamma=10, temperature=50):
        super(CDSNNL, self).__init__()
        self.delta = delta
        self.temperature = temperature
        self.gamma = gamma
        self.cd = chamfer_dist_sep

    def forward(self, reps, pc):
        bs = pc.shape[0]
        n_points = pc.shape[1]
        stacked_pc = pc.repeat(bs, 1, 1)
        stacked_tensors = pc.unsqueeze(1).repeat(1, bs, 1, 1).view(bs * bs, n_points, 3)

        out = self.cd(stacked_pc, stacked_tensors)

        sim = out.view(bs, bs, 1).reshape(bs, bs)
        smap = (sim < self.delta).bool().fill_diagonal_(0)
        sim = 1 - sim * self.gamma
        sim = torch.clamp(sim, min=0)
        sim = sim[smap]

        reps = F.normalize(reps, dim=1)
        similarities = torch.matmul(reps, reps.T) / self.temperature
        positive_similarities = similarities[smap]
        _pexp = torch.exp(positive_similarities)
        weighted_pexp = _pexp * sim
        pexp = torch.mean(weighted_pexp)
        pexp = torch.where(torch.isnan(pexp), torch.tensor(1e-4), pexp)
        pexp = torch.clamp(pexp, min=1e-4)
        texp = torch.sum(torch.exp(similarities))
        out = -torch.log(pexp / texp)
        return out


class CDMSL(nn.Module):
    def __init__(self, chamfer_dist_sep, delta=0.03, margin=0.001):
        super(CDMSL, self).__init__()
        self.thresh = 0.001
        self.margin = margin
        self.scale_pos = 2
        self.scale_neg = 50
        self.delta = delta
        self.cd = chamfer_dist_sep

    def pairwise_chamfer_distance(self, pc):
        bs = pc.shape[0]
        n_points = pc.shape[1]
        stacked_pc = pc.repeat(bs, 1, 1)
        stacked_tensors = pc.unsqueeze(1).repeat(1, bs, 1, 1).view(bs * bs, n_points, 3)

        out = self.cd(stacked_pc, stacked_tensors)
        return out

    def forward(self, features, point_clouds):
        # point_clouds = [batch, n_points, 3] tensor of point clouds
        # features = [batch, feature_dim] tensor of features

        batch_size = point_clouds.size(0)
        sim_mat = self.pairwise_chamfer_distance(point_clouds)
        sim_mat = sim_mat.view(batch_size, batch_size, 1).reshape(
            batch_size, batch_size
        )

        pos_mask = sim_mat < self.delta
        neg_mask = sim_mat >= self.delta
        pos_mask = pos_mask.fill_diagonal_(0)
        neg_mask = neg_mask.fill_diagonal_(0)

        pos_pair = sim_mat[pos_mask]
        neg_pair = sim_mat[neg_mask]

        if len(neg_pair) < 1 or len(pos_pair) < 1:
            return torch.zeros([], requires_grad=True)

        pos_pair = pos_pair[pos_pair - self.margin < torch.max(neg_pair)]
        neg_pair = neg_pair[neg_pair + self.margin > torch.min(pos_pair)]

        if len(neg_pair) < 1 or len(pos_pair) < 1:
            return torch.zeros([], requires_grad=True)

        # Weighting step
        pos_loss = (
            1.0
            / self.scale_pos
            * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
            )
        )
        neg_loss = (
            1.0
            / self.scale_neg
            * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
            )
        )

        loss = pos_loss + neg_loss
        loss.requires_grad = True
        return loss
