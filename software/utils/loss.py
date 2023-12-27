import torch.nn as nn
import torch
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.task_loss = nn.CrossEntropyLoss().to(device="cuda:0")
        self.sparsity_loss = False

    def forward(self, output, labels, model):
        loss = self.task_loss(output, labels)
        return loss
