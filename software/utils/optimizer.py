import torch
import json


class Optimizer:
    def __init__(self, args, params_to_update):
        lr = args.lr
        if "optim_file" not in args:
            self.optimizer = torch.optim.Adam(params_to_update, lr=lr)
            self.use_scheduler = False
        else:
            self.parse_cfg(args.optim_file)
            self.build_optim(params_to_update, lr)
            self.build_scheduler()

    def build_optim(self, params_to_update, lr):
        if self.optim_cfg["type"] == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params_to_update, lr=lr, momentum=self.optim_cfg["momentum"],
                                            weight_decay=self.optim_cfg["weight_decay"])
        elif self.optim_cfg["type"] == 'adam':
            self.optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=self.optim_cfg["weight_decay"])
        elif self.optim_cfg["type"] == 'sgd':
            self.optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=self.optim_cfg["momentum"],
                                        weight_decay=self.optim_cfg["weight_decay"])
        else:
            raise ValueError

    def parse_cfg(self, cfg_file):
        with open(cfg_file, 'r') as cfg:
            cfg = json.load(cfg)
            self.optim_cfg = cfg["optim"]
            if "scheduler" in cfg:
                self.use_scheduler = True
                self.scheduler_cfg = cfg["scheduler"]
            else:
                self.use_scheduler = False

    def build_scheduler(self):
        if self.scheduler_cfg["type"] == "step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_cfg["milestones"],
                                                             gamma=self.scheduler_cfg["gamma"])
        elif self.scheduler_cfg["type"] == "cosine_annealing":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.scheduler_cfg["T_max"])
        elif self.scheduler_cfg["type"] == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.scheduler_cfg["gamma"])
        else:
            raise ValueError

    def schedule_step(self, epoch):
        if self.use_scheduler:
            self.scheduler.step(epoch)


if __name__ == '__main__':
    import argparse
    import torchvision
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim_file", default="../configs/optim_cfg/rmsprop_exp.json")
    parser.add_argument("--lr", default=0.01)
    args = parser.parse_args()
    model = torchvision.models.mobilenet_v2()
    optim = Optimizer(args, model.parameters())
    for epoch in range(200):
        optim.schedule_step(epoch)
        print('{}: {}'.format(epoch, optim.optimizer.param_groups[0]['lr']))