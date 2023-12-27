import torch.nn as nn
import MinkowskiEngine as ME
import os
import numpy as np
from .mobilenet_settings import get_config, get_MNIST_config, get_Roshambo_config
from .drop_utils import DropClass
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor

save_coordinate = False
coord_batch_idx, coord_res_idx = 0, 0
if save_coordinate:
    os.makedirs("coordinates", exist_ok=True)


class InvertedResidualBlockME(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride=1, dimension=2, drop_config=[],
                 relu=ME.MinkowskiReLU6):
        super(InvertedResidualBlockME, self).__init__()

        hidden_channels = int(round(in_channels * expand_ratio))

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
        self.use_drop = False
        if drop_config and self.stride == 2:
            self.use_drop = True
            self.drop = DropClass(type=drop_config[0], drop_config=drop_config[1])

    def forward(self, inp):
        x = inp
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.mask_forward(x, mask)

        x = self.conv2(x)
        if save_coordinate:
            if self.conv2.kernel_generator.kernel_stride[0] == 2:
                global coord_batch_idx, coord_res_idx
                np.save(f"coordinates/{coord_batch_idx}_{coord_res_idx}.npy", x.C.cpu().numpy())
                coord_res_idx += 1

        x = self.bn2(x)
        x = self.relu2(x)
        if self.use_drop:
            x, mask = self.drop(x)
        # else:
        #     x = self.mask_forward(x, mask)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.mask_forward(x, mask)

        if self.use_residual:
            x = x + identity

        return x

    def mask_forward(self, x, mask):
        if mask is None:
            return x
        else:
            return SparseTensor(
                x.F * mask.unsqueeze(dim=1).expand_as(x.F),
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )


class MobileNetV2ME(nn.Module):
    def __init__(self, in_channels, num_classes, width_mult=1.0, MNIST=False, remove_depth=0, drop_config=[],
                 relu="relu6", model_type="base"):
        super(MobileNetV2ME, self).__init__()

        self.relu = ME.MinkowskiReLU if relu == "relu" else ME.MinkowskiReLU6

        input_channels = 32
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width_mult = width_mult

        final_dim = 1280
        if model_type == "Roshambo_cut1stage":
            inverted_residual_setting = [
                # t, c, n, s, drop
                [1, 16, 1, 1, 0],
                [6, 24, 2, 2, 0],
                [6, 32, 3, 2, 0],
                [6, 64, 4, 2, 0],
                [6, 96, 3, 1, 0],
            ]
        elif model_type == "NMNIST_cut2stage":
            inverted_residual_setting = [
                # t, c, n, s, drop
                [1, 16, 1, 1, 0],
                [6, 24, 2, 2, 0],
                [6, 32, 3, 2, 0],
            ]
        elif MNIST:
            inverted_residual_setting = get_MNIST_config(remove_depth, model_type, drop_config)
            final_dim = 128
        elif model_type == "roshambo":
            inverted_residual_setting = get_roshambo_config(remove_depth, model_type, drop_config)
            final_dim = 96
            input_channels = 24
        else:
            inverted_residual_setting, input_channels, final_dim, self.drop_before_block = get_config(remove_depth, model_type, drop_config)
            # final_dim = 1280
        final_input = inverted_residual_setting[-1][1]

        self.conv1 = ME.MinkowskiConvolution(
            in_channels,
            int(input_channels * width_mult),
            kernel_size=3,
            stride=2,
            dimension=2
        )
        self.bn1 = ME.MinkowskiBatchNorm(int(input_channels * width_mult))
        self.relu1 = self.relu(inplace=True)

        blocks = []
        self.drop_blocks = []
        
        input_channels = int(input_channels * width_mult)
        for t, c, n, s, dr in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if self.drop_before_block[0]:
                    self.drop_blocks.append(DropClass(type=self.drop_before_block[1][0],
                                                      drop_config=self.drop_before_block[1][1]).cuda())
                    dr = 0
                stride = s if i == 0 else 1
                blocks.append(InvertedResidualBlockME(
                    input_channels, output_channel, stride=stride, expand_ratio=t, drop_config=dr))
                input_channels = output_channel
        
        self.blocks = nn.Sequential(*blocks)

        self.conv8 = ME.MinkowskiConvolution(
            int(final_input * width_mult),
            int(final_dim * width_mult),
            kernel_size=1,
            stride=1,
            dimension=2
        )
        self.bn8 = ME.MinkowskiBatchNorm(int(final_dim * width_mult))
        self.relu8 = self.relu(inplace=True)

        self.pool = ME.MinkowskiGlobalAvgPooling()
        self.dp = ME.MinkowskiDropout()

        self.fc = ME.MinkowskiLinear(int(final_dim * width_mult), self.num_classes)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.named_modules():
            if isinstance(m, ME.MinkowskiConvolution) or isinstance(m, ME.MinkowskiChannelwiseConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x):
        if save_coordinate:
            global coord_batch_idx, coord_res_idx
            coord_batch_idx += 1
            np.save(f"coordinates/{coord_batch_idx}_1.npy", x.C.cpu().numpy())

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # (x, masks) = self.blocks((x, None))
        for idx, block in enumerate(self.blocks):
            if self.drop_before_block[0]:
                x, _ = self.drop_blocks[idx](x)
            x = block(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.pool(x)
        x = self.dp(x)
        x = self.fc(x)

        return x.F

    def _make_block(self, in_channels, out_channels, t, s):
        layers = []
        layers.append(InvertedResidualBlockME(
            in_channels,
            out_channels,
            stride=s,
            expansion_factor=t,
            dimension=2
        ))

        return nn.Sequential(*layers)