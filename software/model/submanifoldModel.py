import torch
import torch.nn as nn
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
import MinkowskiEngine as ME
import json


class CNN_RNN_Submanifold(nn.Module):
    """
        A baseline eye tracking which uses CNN + GRU to predict the pupil center coordinate
    """
    def __init__(self, args, dimension=2):
        super().__init__()
        in_channels, conv1_channel, conv2_channel, conv3_channel = 3, 16, 32, 64
        self.args = args
        self.device = args.device
        self.dimension = dimension

        self.input_channel = args.n_time_bins
        self.parse_cfg(args.model_cfg)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            conv1_channel,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )
        self.bn1 = ME.MinkowskiBatchNorm(conv1_channel)
        self.relu1 = ME.MinkowskiReLU6(inplace=True)

        self.conv2 = ME.MinkowskiConvolution(
            conv1_channel,
            conv2_channel,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )
        self.bn2 = ME.MinkowskiBatchNorm(conv2_channel)
        self.relu2 = ME.MinkowskiReLU6(inplace=True)

        self.conv3 = ME.MinkowskiConvolution(
            conv2_channel,
            conv3_channel,
            kernel_size=3,
            stride=1,
            dimension=dimension,
        )
        self.bn3 = ME.MinkowskiBatchNorm(conv3_channel)
        self.relu3 = ME.MinkowskiReLU6(inplace=True)

        # self.pool = ME.MinkowskiGlobalAvgPooling()
        # self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 2)


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
        modules.append(ME.MinkowskiConvolution(
            input_channel,
            cfg["filters"],
            kernel_size=cfg["kernel_size"],
            stride=cfg["strides"],
            dimension=self.dimension
        ))
        if "BN" in cfg and cfg["BN"]:
            modules.append(ME.MinkowskiBatchNorm(cfg["filters"]))
        if "activation" in cfg:
            if cfg["activation"] == "relu":
                modules.append(ME.MinkowskiReLU6(inplace=True))
            else:
                raise NotImplementedError
        return modules

    def parse_pooling(self):
        if self.pooling_cfg["type"] == "max":
            self.pool = ME.MinkowskiMaxPooling(kernel_size=self.pooling_cfg["kernel_size"],
                                               stride=self.pooling_cfg["stride"], dimension=self.dimension)
            # self.pool = nn.MaxPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif self.pooling_cfg["type"] == "avg":
            self.pool = ME.MinkowskiAvgPooling(kernel_size=self.pooling_cfg["kernel_size"],
                                               stride=self.pooling_cfg["stride"], dimension=self.dimension)
            # self.pool = nn.AvgPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif self.pooling_cfg["type"] == "global_avg":
            self.pool = ME.MinkowskiGlobalAvgPooling()
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
        x = x.permute(0, 2, 3, 1)

        coord, feat = self.dense_to_sparse(x)
        x = SparseTensor(features=feat.contiguous(), coordinates=coord.int().contiguous(), device=self.device)

        for module in self.convs:
            x = module(x)
        x = self.pool(x)

        x = x.F.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        # output shape of x is (batch_size, seq_len, hidden_size)

        x = self.fc(x)
        # output is of shape (batch_size, seq_len, 2)
        return x

    def dense_to_sparse(self, dense):
        non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense[select_indices], dim=-2)
        return non_zero_indices, features