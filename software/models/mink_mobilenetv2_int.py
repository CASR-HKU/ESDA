import torch.nn as nn
import MinkowskiEngine as ME
import torch
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from .mobilenet_settings import get_config, get_MNIST_config


def conv3x3_dw(out_channel, stride):
    return ME.MinkowskiChannelwiseConvolution(out_channel, kernel_size=3, stride=stride, dimension=2).cuda()


def conv1x1(input_channel, out_channel):
    return ME.MinkowskiConvolution(input_channel, out_channel, kernel_size=1, stride=1, dimension=2).cuda()


def conv3x3(input_channel, out_channel):
    return ME.MinkowskiConvolution(input_channel, out_channel, kernel_size=3, stride=1, dimension=2).cuda()


class InvertedResidualBlockMEInt(nn.Module):
    def __init__(self, model=None, bit_shift=8, stride=1, use_residual=False):
        super(InvertedResidualBlockMEInt, self).__init__()

        self.bit_shift = bit_shift
        self.stride = stride
        self.relu = ME.MinkowskiReLU6(inplace=True)
        self.use_residual = use_residual
        if model is not None:
            self.load_weight(model)
        else:
            self.init_None()

    def load_weight(self, model):
        self.register_buffer("conv1_weight_int",  model.conv1.weight_integer)
        self.register_buffer("conv1_bias_int", model.conv1.bias_integer)
        self.register_buffer("conv2_weight_int", model.conv2.weight_integer)
        self.register_buffer("conv2_bias_int", model.conv2.bias_integer)
        self.register_buffer("conv3_weight_int", model.conv3.weight_integer)
        self.register_buffer("conv3_bias_int", model.conv3.bias_integer)

        self.register_buffer("conv1_scaling", torch.tensor(model.quant_act.act_scaling_factor) *
                             model.conv1.convbn_scaling_factor/torch.tensor(model.quant_act1.act_scaling_factor))
        self.register_buffer("conv2_scaling", torch.tensor(model.quant_act1.act_scaling_factor) *
                             model.conv2.convbn_scaling_factor/torch.tensor(model.quant_act2.act_scaling_factor))
        self.register_buffer("conv3_scaling", torch.tensor(model.quant_act2.act_scaling_factor) *
                             model.conv3.convbn_scaling_factor/torch.tensor(model.quant_act_int32.act_scaling_factor))

        if self.use_residual:
            self.register_buffer("main_scaling", torch.tensor(model.quant_act1.act_scaling_factor) /
                                 torch.tensor(model.quant_act_int32.act_scaling_factor))
            self.register_buffer("residual_scaling", torch.tensor(model.quant_act.act_scaling_factor) /
                                 torch.tensor(model.quant_act_int32.act_scaling_factor))

    def init_None(self):
        self.register_buffer("conv1_weight_int",  torch.zeros(1))
        self.register_buffer("conv1_bias_int", torch.zeros(1))
        self.register_buffer("conv2_weight_int", torch.zeros(1))
        self.register_buffer("conv2_bias_int", torch.zeros(1))
        self.register_buffer("conv3_weight_int", torch.zeros(1))
        self.register_buffer("conv3_bias_int", torch.zeros(1))

        self.register_buffer("conv1_scaling", torch.zeros(1))
        self.register_buffer("conv2_scaling", torch.zeros(1))
        self.register_buffer("conv3_scaling", torch.zeros(1))

        if self.use_residual:
            self.register_buffer("main_scaling", torch.zeros(1))
            self.register_buffer("residual_scaling", torch.zeros(1))

    def forward(self, x):
        identity = x
        x = conv1x1(self.conv1_weight_int.shape[0], self.conv1_weight_int.shape[1])\
            (x, None, (self.conv1_weight_int, self.conv1_bias_int,
                       torch.round(self.conv1_scaling) * (2**self.bit_shift)))
        x = SparseTensor(
            torch.clamp(x, -(2**(self.bit_shift-1)), 2**(self.bit_shift-1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.relu(x)

        x = conv3x3_dw(self.conv2_weight_int.shape[1], self.stride)\
            (x, None, (self.conv2_weight_int, self.conv2_bias_int, torch.round(self.conv2_scaling) * (2**self.bit_shift)))
        x = SparseTensor(
            torch.clamp(x, -(2**(self.bit_shift-1)), 2**(self.bit_shift-1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.relu(x)

        x = conv1x1(self.conv3_weight_int.shape[0], self.conv3_weight_int.shape[1])\
            (x, None, (self.conv3_weight_int, self.conv3_bias_int,
                       torch.round(self.conv3_scaling) * (2**self.bit_shift)))
        x = torch.clamp(x, -(2**(self.bit_shift-1)), 2**(self.bit_shift-1))

        if self.use_residual:
            out = torch.round(self.main_scaling) * (2**self.bit_shift) * x.F + \
                torch.round(self.residual_scaling) * (2**self.bit_shift) * identity.F
            x = SparseTensor(
                out,
                coordinate_map_key=x.coordinate_map_key,
                coordinate_manager=x.coordinate_manager,
            )
        return x


class MobileNetV2MEInt(nn.Module):
    def __init__(self, in_channels, num_classes, width_mult=1.0, MNIST=False, bit_shift=8, remove_depth=0):
        super(MobileNetV2MEInt, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.bit_shift = bit_shift
        self.MNIST = MNIST
        self.inverted_residual_setting = get_MNIST_config(remove_depth) if MNIST else get_config(remove_depth)

        self.relu1 = ME.MinkowskiReLU6(inplace=True)

    def init_weight_None(self):
        self.register_buffer('conv1_scaling', torch.zeros(1))
        self.register_buffer('conv1_weight', torch.zeros(1))
        self.register_buffer('conv1_bias', torch.zeros(1))
        self.register_buffer('conv8_weight', torch.zeros(1))
        self.register_buffer('conv8_bias', torch.zeros(1))
        self.register_buffer("conv8_scaling", torch.zeros(1))
        self.register_buffer("quant_act_output", torch.zeros(1))

    def load_weight(self, model):
        self.register_buffer('conv1_scaling', torch.ones(1).cuda()*model.conv1.convbn_scaling_factor/
                             torch.tensor(model.quant_act_after_first_block.act_scaling_factor) )
        self.register_buffer('conv1_weight', model.conv1.weight_integer)
        self.register_buffer('conv1_bias', model.conv1.bias_integer)
        self.register_buffer('conv8_weight', model.conv8.weight_integer)
        self.register_buffer('conv8_bias', model.conv8.bias_integer)
        self.register_buffer("conv8_scaling", torch.tensor(model.quant_act_before_final_block.act_scaling_factor) *
                             model.conv8.convbn_scaling_factor /
                             torch.tensor(model.quant_act_before_final_block.act_scaling_factor))
        self.register_buffer("quant_act_output", torch.tensor(model.quant_act_output.act_scaling_factor))

    def init_weights(self, model=None):
        self.relu = ME.MinkowskiReLU6(inplace=True)
        self.pool = ME.MinkowskiGlobalAvgPooling()
        self.fc = ME.MinkowskiLinear(int(1280 * self.width_mult), self.num_classes)

        if model is None:
            self.init_weight_None()
        else:
            self.load_weight(model)

        input_channels = int(self.in_channels * self.width_mult)
        blocks = []
        block_idx = 0
        for t, c, n, s, dr in self.inverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                use_residual = (input_channels == output_channel and s == 1)
                if model is None:
                    blocks.append(InvertedResidualBlockMEInt(None, self.bit_shift, s, use_residual))
                else:
                    blocks.append(InvertedResidualBlockMEInt(model.blocks[block_idx], self.bit_shift, s, use_residual))
                input_channels = output_channel
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = conv3x3(self.conv1_weight.shape[1], self.conv1_weight.shape[2]) \
                (x, None, (self.conv1_weight, self.conv1_bias, torch.round(self.conv1_scaling.data) * (2**self.bit_shift)))
        x = SparseTensor(
            torch.clamp(x, -(2**(self.bit_shift-1)), 2**(self.bit_shift-1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.relu(x)

        x = self.blocks(x)

        x = conv1x1(self.conv8_weight.shape[0], self.conv8_weight.shape[1]) \
                (x, None, (self.conv8_weight, self.conv8_bias, torch.round(self.conv8_scaling.data) * (2**self.bit_shift)))
        x = SparseTensor(
            torch.clamp(x, -(2**(self.bit_shift-1)), 2**(self.bit_shift-1)),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x = self.relu(x)

        x = self.pool(x)
        x = self.fc(x)

        return x.F
