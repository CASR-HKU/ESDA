from torch import nn
from typing import Callable, Any, Optional, List
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
import MinkowskiEngine as ME
import json
from .utils import dense_to_sparse, RandomDrop
import sys


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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        dimension: int = 2
    ) -> None:
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.kernel_size = kernel_size
        if norm_layer is None:
            norm_layer = ME.MinkowskiBatchNorm
        if activation_layer is None:
            activation_layer = ME.MinkowskiReLU6
        self.stride = stride
        self.out_channels = out_planes
        self.conv = ME.MinkowskiConvolution(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            dimension=dimension,
        )
        self.norm = norm_layer(out_planes)
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class InvertedResidualBlockME(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride=1, dimension=2, relu=ME.MinkowskiReLU6,
                 drop_config=None):
        super(InvertedResidualBlockME, self).__init__()

        hidden_channels = int(round(in_channels * expand_ratio))
        conv1_drop, conv2_drop, conv3_drop = drop_config
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        if conv1_drop:
            self.drop1 = RandomDrop((conv1_drop, False))
        if conv2_drop:
            self.drop2 = RandomDrop((conv2_drop, False))
        if conv3_drop:
            self.drop3 = RandomDrop((conv3_drop, False))

        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.bn1 = ME.MinkowskiBatchNorm(hidden_channels)
        self.relu1 = relu(inplace=True)

        self.conv2 = ME.MinkowskiChannelwiseConvolution(
            hidden_channels,
            kernel_size=3,
            stride=stride,
            dimension=dimension
        )

        self.bn2 = ME.MinkowskiBatchNorm(hidden_channels)
        self.relu2 = relu(inplace=True)

        self.conv3 = ME.MinkowskiConvolution(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.bn3 = ME.MinkowskiBatchNorm(out_channels)

        self.use_residual = (in_channels == out_channels and stride == 1)
        self.stride = stride


    def forward(self, x):
        if isinstance(x, tuple):
            save_json = True
            x, struct, res_idx, b_idx, kernel_sparsity, activation_sparsity = x
            act_sparsity = []
            tensor_strides = []
            ks = kernel_sparsity[res_idx]
        else:
            save_json = False

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if save_json:
            act_sparsity.append(activation_sparsity[res_idx])
            tensor_strides.append(x.tensor_stride[0])
        if hasattr(self, "drop1"):
            x, _ = self.drop1(x)

        x = self.conv2(x)
        if self.stride == 2 and save_json:
            res_idx += 1

        x = self.bn2(x)
        x = self.relu2(x)
        if save_json:
            act_sparsity.append(activation_sparsity[res_idx])
            tensor_strides.append(x.tensor_stride[0])

        if hasattr(self, "drop2"):
            x, _ = self.drop2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if save_json:
            act_sparsity.append(activation_sparsity[res_idx])
            tensor_strides.append(x.tensor_stride[0])
            conv_json = {
                "name": "block.{}".format(b_idx),
                "type": "block",
                "residual": self.use_residual,
                "stride": self.stride,
                "tensor_stride": tensor_strides,
                "channels": [self.in_channels, self.hidden_channels, self.out_channels],
                "sparsity": act_sparsity,
                "kernel_sparsity": ks,
            }
            b_idx += 1
            struct["layers"].append(conv_json)

        if hasattr(self, "drop3"):
            x, _ = self.drop3(x)

        if self.use_residual:
            x = x + identity

        if save_json:
            return (x, struct, res_idx, b_idx, kernel_sparsity, activation_sparsity)
        else:
            return x


class MobileNetSubmanifold(nn.Module):
    def __init__(
        self, args, round_nearest: int = 8, sample_channel = 3, num_classes: int = 2
    ) -> None:
        super(MobileNetSubmanifold, self).__init__()

        self.device = args.device
        block = InvertedResidualBlockME
        norm_layer = ME.MinkowskiBatchNorm
        self.use_heatmap = False
        self.sample_channel = sample_channel
        self.num_classes = num_classes

        with open(args.model_cfg, 'r') as ft:
            cfg = json.load(ft)

        width_mult = 1.0
        if "width_mult" in cfg:
            width_mult = cfg["width_mult"]

        input_channel = cfg["input_channel"]
        last_channel = cfg["last_channel"]
        inverted_residual_setting = cfg["backbone"]
        self.inverted_residual_setting = inverted_residual_setting

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        drop_config = self.parse_drop(args, inverted_residual_setting)

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(sample_channel, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        block_idx = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(
                    input_channel, output_channel, stride=stride, expand_ratio=t, drop_config=drop_config[block_idx]))
                input_channel = output_channel
                block_idx += 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        if "heatmap" in cfg and cfg["heatmap"]:
            self.use_heatmap = True
            upsample_nums = len([1 for block in inverted_residual_setting if block[-1] > 1])
            upsample_blocks = []
            in_channel = self.last_channel
            from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolutionTranspose
            for idx in range(upsample_nums-1):
                upsample_blocks.append(MinkowskiConvolutionTranspose(in_channels=in_channel, out_channels=in_channel//2,
                                                          kernel_size=3, stride=2, dimension=2))
                in_channel = in_channel // 2
            upsample_blocks.append(MinkowskiConvolutionTranspose(in_channels=in_channel, out_channels=1, kernel_size=3, stride=2, dimension=2))
            self.upsample_blocks = nn.Sequential(*upsample_blocks)
        else:
            self.use_heatmap = False
            self.parse_pooling(cfg["pool"])
            self.parse_rnn(cfg["rnn"], self.last_channel)
            self.classifier = nn.Sequential(
                # nn.Dropout(0.2),
                nn.Linear(self.fc_in, num_classes),
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

    def parse_drop(self, args, backbone):
        block_nums = sum([i[-2] for i in backbone])
        if "drop_config" not in args:
            return [[0, 0, 0] for __ in range(block_nums)]
        with open(args.drop_config, 'r') as cfg:
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


    def parse_rnn(self, rnn_cfg, rnn_in_channel):
        self.fc_in = rnn_cfg["units"]
        if rnn_cfg["type"] == "gru":
            self.rnn = nn.GRU(input_size=rnn_in_channel, hidden_size=rnn_cfg["units"], num_layers=rnn_cfg["num_layers"],
                              batch_first=True)
        elif rnn_cfg["type"] == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_in_channel, hidden_size=rnn_cfg["units"],
                               num_layers=rnn_cfg["num_layers"], batch_first=True)
        else:
            raise NotImplementedError

    def parse_pooling(self, pooling_cfg):
        if pooling_cfg["type"] == "max":
            self.pool = ME.MinkowskiMaxPooling(kernel_size=pooling_cfg["kernel_size"],
                                               stride=pooling_cfg["stride"], dimension=self.dimension)
            # self.pool = nn.MaxPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif pooling_cfg["type"] == "avg":
            self.pool = ME.MinkowskiAvgPooling(kernel_size=pooling_cfg["kernel_size"],
                                               stride=pooling_cfg["stride"], dimension=self.dimension)
            # self.pool = nn.AvgPool2d(kernel_size=self.pooling_cfg["kernel_size"], stride=self.pooling_cfg["stride"])
        elif pooling_cfg["type"] == "global_avg":
            self.pool = ME.MinkowskiGlobalAvgPooling()
        else:
            raise NotImplementedError

    def forward(self, x, name="", size=""):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        x = x.permute(0, 2, 3, 1)

        coord, feat = dense_to_sparse(x)
        x = SparseTensor(features=feat.contiguous(), coordinates=coord.int().contiguous(), device=self.device)

        save_json = True if name and size else False
        resolution_idx = 0
        if save_json:
            import json
            from .sparsity import get_sparsity
            activation_sparsity , kernel_sparsity = get_sparsity(name)
            json_file = open(self.json_file, "w")
            struct = {
                "name": "MobileNetV2",
                "dataset": name,
                "input_shape": size,
                "input_sparsity": activation_sparsity[resolution_idx],
                "settings": self.inverted_residual_setting,
                "param": self.param,
                "layers": []
            }
        x = self.features[0](x)
        if save_json:
            resolution_idx += 1
            conv_json = {
                "name": "conv1",
                "type": "conv",
                "residual": False,
                "stride": self.features[0].conv.kernel_generator.kernel_stride[0],
                "tensor_stride": [int(x.tensor_stride[0]/2), x.tensor_stride[1]],
                "channels": [self.features[0].conv.in_channels, self.features[0].conv.out_channels],
                "sparsity": activation_sparsity[resolution_idx],
                "kernel_sparsity": kernel_sparsity[resolution_idx],
            }
            struct["layers"].append(conv_json)

        # x = self.features[1:-1](x)
            (x, struct, resolution_idx, block_idx, kernel_sparsity, activation_sparsity) = \
                self.features[1:-1]((x, struct, resolution_idx, 0, kernel_sparsity, activation_sparsity))
        else:
            x = self.features[1:-1](x)

        x = self.features[-1](x)
        if save_json:
            conv_json = {
                "name": "conv8",
                "type": "conv",
                "residual": False,
                "stride": self.features[-1].conv.kernel_generator.kernel_stride[0],
                "tensor_stride": [x.tensor_stride[0], x.tensor_stride[0]],
                "channels": [self.features[-1].conv.in_channels, self.features[-1].conv.out_channels],
                "sparsity": activation_sparsity[resolution_idx],
                "kernel_sparsity": kernel_sparsity[resolution_idx],
            }
            struct["layers"].append(conv_json)

        x = self.pool(x)
        if save_json:
            conv_json = {
                "name": "fc",
                "type": "linear",
                "residual": False,
                "stride": 1,
                "tensor_stride": [x.tensor_stride[0], x.tensor_stride[0]],
                "channels": [self.features[-1].out_channels, self.num_classes],
                "sparsity": 0,
                "kernel_sparsity": [0 for _ in range(9)],
            }
            struct["layers"].append(conv_json)
            json.dump(struct, json_file, indent=4)
            # sys.exit(0)


        x = x.F.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        return x
