import torch
import json
import torch.nn as nn


class CNN_RNN(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args):
        super().__init__()
        self.use_heatmap = False
        self.args = args
        self.input_channel = args.n_time_bins
        self.parse_cfg(args.model_cfg)
        self.fc = nn.Linear(self.fc_in, 2)

    def parse_cfg(self, cfg_path):
        with open(cfg_path, 'r') as ft:
            cfg = json.load(ft)
        self.convs = []
        input_channel = self.input_channel
        for idx, conv_cfg in cfg["conv"].items():
            self.convs += self.parse_conv(conv_cfg, input_channel)
            input_channel = conv_cfg["filters"]

        self.convs = nn.ModuleList(self.convs)
        self.pooling_cfg = cfg["pooling"]
        self.parse_pooling()
        self.use_rnn = False
        if "rnn" in cfg:
            self.use_rnn = True
            self.rnn_cfg = cfg["rnn"]
            self.parse_rnn()

    def parse_conv(self, cfg, input_channel):
        modules = []
        modules.append(nn.Conv2d(input_channel, cfg["filters"], kernel_size=cfg["kernel_size"], stride=cfg["strides"],
                         padding=1))
        if "BN" in cfg and cfg["BN"]:
            modules.append(nn.BatchNorm2d(cfg["filters"]))
        if "activation" in cfg:
            if cfg["activation"] == "relu":
                modules.append(nn.ReLU())
            else:
                raise NotImplementedError
        return modules

    def parse_pooling(self):
        if self.pooling_cfg["type"] == "max":
            self.pool = nn.MaxPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif self.pooling_cfg["type"] == "avg":
            self.pool = nn.AvgPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif self.pooling_cfg["type"] == "global_avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise NotImplementedError

    def parse_rnn(self):
        self.fc_in = self.rnn_cfg["units"]
        if self.rnn_cfg["type"] == "gru":
            self.rnn = nn.GRU(input_size=self.rnn_cfg["input_size"], hidden_size=self.rnn_cfg["units"],
                              num_layers=self.rnn_cfg["num_layers"], batch_first=True)
        elif self.rnn_cfg["type"] == "lstm":
            self.rnn = nn.LSTM(input_size=self.rnn_cfg["input_size"], hidden_size=self.rnn_cfg["units"],
                               num_layers=self.rnn_cfg["num_layers"], batch_first=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        # input is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        for module in self.convs:
            x = module(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        if self.use_rnn:
            x, _ = self.rnn(x)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x
