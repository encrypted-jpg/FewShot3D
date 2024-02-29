import torch
import torch.nn.functional as F


class BaseSNNLoss:
    def __init__(self, temperature=50):
        self.temperature = temperature

    def __call__(self, reps, labels):
        reps = F.normalize(reps, dim=1)

        positive_pairs = labels.unsqueeze(1) == labels.unsqueeze(0)
        positive_pairs.fill_diagonal_(False)  # Remove self-pairs

        similarities = torch.matmul(reps, reps.T)/self.temperature
        positive_similarities = similarities[positive_pairs]

        pexp = torch.mean(torch.exp(positive_similarities))
        texp = torch.sum(torch.exp(similarities))

        out = -torch.log(pexp/texp)
        return out
