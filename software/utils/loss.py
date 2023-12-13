import torch.nn as nn
import torch
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, settings, lamb):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss().to(device="cuda:0")
        self.lamb = lamb
        self.sparsity_loss = False
        if settings.drop_config and settings.drop_config["type"] == "abs_sum":
            self.sparsity_loss = True

        self.sparsity_type = "threshold"
        if hasattr(settings, "drop_config"):
            if "sparsity_type" in settings.drop_config:
                self.sparsity_type = settings.drop_config["sparsity_type"]
        assert self.sparsity_type in ["threshold", "prune_ratio"], "Sparsity type not implemented"
        if self.sparsity_type == "prune_ratio":
            self.budget = settings.drop_config["budget"]
            if isinstance(self.budget, float):
                self.budget = [self.budget] * 4
            self.unlimited_lower = False
            if "unlimited_lower" in settings.drop_config:
                self.unlimited_lower = settings.drop_config["unlimited_lower"]

    def forward(self, output, labels, model):
        loss = self.task_loss(output, labels)
        if self.sparsity_loss:
            if self.sparsity_type == "threshold":
                for name, param in model.named_parameters():
                    if 'threshold' in name:
                        loss += self.lamb * torch.sum((1 - param) ** 2)
            elif self.sparsity_type == "prune_ratio":
                cnt = 0
                for n, m in model.named_modules():
                    if hasattr(m, "skip_ratio"):
                        loss += (F.relu(self.budget[cnt] - m.skip_ratio))**2 * self.lamb
                        if not self.unlimited_lower:
                            loss += (F.relu(m.skip_ratio))**2 * self.lamb
                        cnt += 1
        return loss
