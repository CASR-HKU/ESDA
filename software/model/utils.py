import torch
import numpy as np
import MinkowskiEngine as ME
from torch import nn


def dense_to_sparse(dense):
    non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense[select_indices], dim=-2)
    return non_zero_indices, features


class RandomDrop(nn.Module):
    def __init__(self, drop_config=[0.5, False]):
        super(RandomDrop, self).__init__()
        self.ratio = drop_config[0]
        self.prune = ME.MinkowskiPruning()
        self.gradually = drop_config[1]
        self.epoch_ratio = 0

    def __call__(self, x):
        bs = x.C[:, 0].unique()
        mask = torch.ones(x.C.shape[0], device=x.device)
        for b_idx in range(len(bs)):
            mask = self.update_sample_mask(b_idx, mask, x.C.cpu())
        return self.prune(x, mask.bool()), None
        # mask = torch.rand(x.shape[0], device=x.device) >= self.drop_ratio
        return SparseTensor(
            x.F * mask.unsqueeze(dim=1).expand_as(x.F),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

    def update_sample_mask(self, b_idx, mask, coord):
        ratio = self.ratio if not self.gradually else self.ratio * self.epoch_ratio
        coord_location = np.where(coord[:, 0] == b_idx)[0]
        target_size = int(len(coord_location) * ratio)
        rm_coord = np.random.choice(coord_location, target_size, replace=False)
        mask[rm_coord] = 0
        return mask