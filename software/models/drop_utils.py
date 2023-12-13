import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor


class DropClass(nn.Module):
    def __init__(self, type, **kwargs):
        super(DropClass, self).__init__()
        if type == "random":
            self.prune = RandomDrop(**kwargs)
        elif type == "abs_sum":
            self.prune = AbsSumDrop(**kwargs)
        elif type == "abs_sum_momentum":
            self.prune = AbsSumDropMomentum(**kwargs)
        elif type == "none":
            self.prune = NoneDrop()
        else:
            raise NotImplementedError("Drop type not implemented")

    def __call__(self, x):
        return self.prune(x)


class NoneDrop(nn.Module):
    def __init__(self):
        super(NoneDrop, self).__init__()
        pass

    def __call__(self, x):
        return x


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


class AbsSumDropMomentum(nn.Module):
    def __init__(self, drop_config):
        super(AbsSumDropMomentum, self).__init__()
        self.threshold = nn.Parameter(0 * torch.ones(1), requires_grad=False)
        self.ratio = drop_config[0]
        self.momentum = 0.99
        self.prune = ME.MinkowskiPruning()
        self.gradually = drop_config[1]
        self.epoch_ratio = 0

    def forward(self, x):
        ratio = self.ratio if not self.gradually else self.ratio * self.epoch_ratio
        abs_avg = torch.abs(x.F).mean(1)
        idx = int(abs_avg.shape[0] * ratio)
        threshold = torch.sort(abs_avg)[0][idx].detach()
        self.skip_ratio = 1 - (abs_avg > self.threshold).sum().item() / abs_avg.shape[0]
        mask = abs_avg > self.threshold
        x = self.prune(x, mask)
        if self.training:
            self.threshold *= self.momentum
            self.threshold += threshold * (1 - self.momentum)
        return x, None


class AbsSumDrop(nn.Module):
    def __init__(self, drop_config):
        super(AbsSumDrop, self).__init__()
        self.threshold = nn.Parameter(drop_config[0] * torch.ones(1))
        self.gt = GreaterThan.apply

    def __call__(self, x):
        abs_avg = torch.abs(x.F).mean(1)
        mask = self.gt(torch.sigmoid(abs_avg - self.threshold) - 0.5 * torch.ones_like(abs_avg))
        self.skip_ratio = 1 - mask.sum() / mask.numel()
        masked_output = SparseTensor(
            x.F * mask.unsqueeze(dim=1).expand_as(x.F),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        return masked_output, mask


class GreaterThan(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        # print("Forward here")
        return torch.Tensor.float(torch.gt(input, torch.zeros_like(input)))

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # print("Backward here")
        return grad_input, None