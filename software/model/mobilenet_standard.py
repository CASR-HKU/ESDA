from torch import nn
import torch
from typing import Callable, Any, Optional, List
from torch.hub import load_state_dict_from_url
import json
from .ConvGRU import ConvGRUCell


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.kernel_size = kernel_size
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        self.stride = stride
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False)
        self.norm = norm_layer(out_planes)
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        drop_config: dict = None,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        self.conv1 = ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer)
        self.conv2 = ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer)
        self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = norm_layer(oup)

        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):

        if self.use_res_connect:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.use_res_connect:
            return x + identity
        else:
            return x


class MobileNetStandard(nn.Module):
    def __init__(
        self, args, round_nearest: int = 8, sample_channel = 3, num_classes: int = 2
    ) -> None:
        super(MobileNetStandard, self).__init__()

        block = InvertedResidual
        norm_layer = nn.BatchNorm2d

        if isinstance(args, dict):
            cfg = args
        else:
            with open(args.model_cfg, 'r') as ft:
                cfg = json.load(ft)

        width_mult = 1.0
        if "width_mult" in cfg:
            width_mult = cfg["width_mult"]

        input_channel = cfg["input_channel"]
        last_channel = cfg["last_channel"]
        inverted_residual_setting = cfg["backbone"]

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        drop_config = self.parse_drop(args, inverted_residual_setting)
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(sample_channel, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        block_id = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer,
                                      drop_config=drop_config[block_id]))
                input_channel = output_channel
                block_id += 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.use_rnn = False
        self.rnn_outsize = self.last_channel
        if "rnn" in cfg:
            self.use_rnn = True
            self.parse_rnn(cfg["rnn"], self.last_channel)

        if "heatmap" in cfg and cfg["heatmap"]:
            self.use_heatmap = True
            upsample_nums = len([1 for block in inverted_residual_setting if block[-1] > 1])
            upsample_blocks = []
            in_channel = self.rnn_outsize
            for idx in range(upsample_nums):
                upsample_blocks.append(torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=in_channel//2,
                                                                kernel_size=3, stride=2))
                upsample_blocks.append(nn.BatchNorm2d(in_channel//2))
                upsample_blocks.append(nn.ReLU6(inplace=True))
                in_channel = in_channel // 2
            # upsample_blocks.append(torch.nn.ConvTranspose2d(in_channels=in_channel, out_channels=1, kernel_size=3,
            #                                                 stride=2))
            self.conv_pred = nn.Conv2d(in_channel, 1, kernel_size=1)
            self.upsample_blocks = nn.Sequential(*upsample_blocks)
        else:
            self.use_heatmap = False
            self.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(self.rnn_outsize, num_classes),
            )

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def parse_rnn(self, rnn_cfg, rnn_in_channel):

        if rnn_cfg["type"] == "gru":
            self.rnn_outsize = rnn_cfg["units"]
            self.rnn = nn.GRU(input_size=rnn_in_channel, hidden_size=rnn_cfg["units"], num_layers=rnn_cfg["num_layers"],
                              batch_first=True)
        elif rnn_cfg["type"] == "lstm":
            self.rnn_outsize = rnn_cfg["units"]
            self.rnn = nn.LSTM(input_size=rnn_in_channel, hidden_size=rnn_cfg["units"],
                               num_layers=rnn_cfg["num_layers"], batch_first=True)
        elif rnn_cfg["type"] == "ConvGRU":
            num_layers, hidden_size = rnn_cfg["num_layers"], rnn_cfg["hidden_size"]
            if isinstance(hidden_size, int):
                hidden_size = [hidden_size] * num_layers
            self.rnn_outsize = hidden_size[-1]
            kernel_size, padding = 3, 1
            rnns = []
            for i in range(num_layers):
                rnns.append(ConvGRUCell(input_dim=rnn_in_channel, hidden_dim=hidden_size[i],
                                        kernel_size=kernel_size, bias=True))
            self.rnn = nn.ModuleList(rnns)
        else:
            raise NotImplementedError

    def parse_drop(self, args, backbone):
        block_nums = sum([i[-2] for i in backbone])
        if "drop_config" not in args:
            return [[0, 0, 0] for __ in range(block_nums)]
        drop_file = args.drop_config
        with open(drop_file, 'r') as cfg:
            cfg = json.load(cfg)
        drop_type = cfg["type"]
        drop_p = cfg["ratio"]
        if drop_type == "all_layer":
            drop_config = [[drop_p, drop_p, drop_p] for __ in range(block_nums)]
        elif drop_type == "all_block":
            drop_config = [[0, 0, drop_p] for __ in range(block_nums)]
        elif drop_type == "all_stage":
            prv_block = 0
            drop_config = [[0, 0, 0] for __ in range(block_nums)]
            for b in backbone:
                drop_config[prv_block] = [0, 0, drop_p]
                prv_block += b[-2]
        elif drop_type == "only_stride2":
            layer_stride = [1 for _ in range(block_nums)]
            prv_layers = 0
            for idx, b in enumerate(backbone):
                if b[-1] == 2:
                    layer_stride[prv_layers] = 2
                prv_layers += b[-2]
            drop_config = []
            for stg_idx in range(block_nums):
                if layer_stride[stg_idx] == 2:
                    drop_config.append([0, drop_p, 0])
                else:
                    drop_config.append([0, 0, 0])
        else:
            raise NotImplementedError
        return drop_config


    def _forward_impl(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        # permute height and width
        x = x.permute(0, 1, 3, 2)

        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        if self.use_heatmap:
            _, c, feat_width, feat_height = x.shape
            if self.use_rnn:
                x = x.view(batch_size, seq_len, c, feat_width, feat_height)
                hidden_state = self.rnn[0].init_hidden(batch_size, feat_width, feat_height).to(x.device)
                for idx in range(len(self.rnn)):
                    rnn_outputs = []
                    for t in range(seq_len):
                        # x, hidden_state = rnn((x, hidden_state))
                        hidden_state = self.rnn[idx](input_tensor=x[:, t, :, :, :], h_cur=hidden_state)
                        rnn_outputs.append(hidden_state)
                        # output_inner.append(h)
                # x = self.rnn((x, hidden_state))
                rnn_outputs = torch.stack(rnn_outputs, dim=1)
                _, rnn_out_channel, rnn_out_width, rnn_out_height = hidden_state.shape
                x = rnn_outputs.view(batch_size * seq_len, rnn_out_channel, rnn_out_width, rnn_out_height)

            x = self.upsample_blocks(x)
            x = torch.nn.functional.interpolate(x, (width, height))
            x = self.conv_pred(x)
            x = x.view(batch_size, seq_len, width, height)
            return x
        else:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(batch_size, seq_len, -1)
            x, _ = self.rnn(x)
            x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

