"""
    Quantized MobileNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""


from models.HAWQ_quant_module.quant_modules import *
import MinkowskiEngine as ME
from .mobilenet_settings import get_config, get_MNIST_config, get_roshambo_config


class Q_LinearBottleneck(nn.Module):
    def __init__(self,
                model,
                in_channels,
                out_channels,
                stride,
                expand_ratio,
                shift_bit=31,
                bias_bit=32,
                drop_config=[],
                **kwargs):
        """
        So-called 'Linear Bottleneck' layer. It is used as a quantized MobileNetV2 unit.
        Parameters:
        ----------
        model : nn.Module
            The pretrained floating-point couterpart of this module with the same structure.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int or tuple/list of 2 int
            Strides of the second convolution layer.
        expansion : bool
            Whether do expansion of channels.
        remove_exp_conv : bool
            Whether to remove expansion convolution.
        """
        super(Q_LinearBottleneck, self).__init__()
        self.use_residual = model.use_residual

        # hidden_channels = in_channels * expand_ratio
        # self.use_exp_conv = (expand_ratio != 1)
        self.activation_func = ME.MinkowskiReLU6(inplace=True)

        # self.quant_act = QuantAct(activation_bit=8, shift_bit=shift_bit)

        # if self.use_exp_conv:
        self.conv1 = QuantBnConv2d(per_channel=True, bias_bit=bias_bit, **kwargs)
        self.conv1.set_param(model.conv1, model.bn1)
        self.quant_act1 = QuantAct(shift_bit=shift_bit)

        self.conv2 = QuantBnConv2d(per_channel=True, bias_bit=bias_bit, **kwargs)
        self.conv2.set_param(model.conv2, model.bn2)
        self.quant_act2 = QuantAct(shift_bit=shift_bit)

        self.conv3 = QuantBnConv2d(per_channel=True, bias_bit=bias_bit, **kwargs)
        self.conv3.set_param(model.conv3, model.bn3)

        self.quant_act_int32 = QuantAct(shift_bit=shift_bit)

    def forward(self, x, scaling_factor_int32=None, mask=None):
        if isinstance(x, tuple):
            x, scaling_factor_int32, mask = x
        if self.use_residual:
            identity = x

        # x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None)
        act_scaling_factor = scaling_factor_int32
        identity_scaling_factor = act_scaling_factor

        # if self.use_exp_conv:
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.activation_func(x)

        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)


        # note that, there is no activation for the last conv
        x, weight_scaling_factor = self.conv3(x, act_scaling_factor)
        # else:
        #     x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
        #     x = self.activation_func(x)
        #     x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)
        #
        #     # note that, there is no activation for the last conv
        #     x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        if self.use_residual:
            x = x + identity
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_scaling_factor, None)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        return x, act_scaling_factor, mask




class Q_MobileNetV2(nn.Module):
    """
    Quantized MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    remove_exp_conv : bool
        Whether to remove expansion convolution.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 model,
                 in_channels,
                 width_mult=1.0,
                 num_classes=1000,
                 conv1_bit=8,
                 MNIST=False,
                 remove_depth=0,
                 shift_bit=31,
                 bias_bit=32,
                 drop_config=[],
                 model_type="base",
                 **kwargs):
        super(Q_MobileNetV2, self).__init__()

        input_channels = 32
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.in_channels = in_channels
        self.MNIST = MNIST
        self.shift_bit = shift_bit
        # self.channels = channels

        self.quant_act_before_first_block = QuantAct(shift_bit=shift_bit)
        self.conv1 = QuantBnConv2d(per_channel=True, weight_bit=conv1_bit, bias_bit=bias_bit, **kwargs)
        self.conv1.set_param(model.conv1, model.bn1)
        self.quant_act_after_first_block = QuantAct(shift_bit=shift_bit)

        self.activation_func = ME.MinkowskiReLU6(inplace=True)
        # self.spatial_size = model.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        # self.inputLayer = scn.InputLayer(dimension=2, spatial_size=self.spatial_size, mode=2)

        # add input quantization
        # self.quant_input = QuantAct()

        self.drop_before_block = [False]
        if MNIST:
            inverted_residual_setting = get_MNIST_config(remove_depth, model_type, drop_config)
            final_dim = 128
        elif model_type == "IniRosh":
            inverted_residual_setting = get_Roshambo_config(remove_depth, model_type, drop_config)
            final_dim = 96
            input_channels = 24
        else:
            inverted_residual_setting, input_channels, final_dim, self.drop_before_block = get_config(remove_depth, model_type, drop_config)
            # final_dim = 1280
        self.inverted_residual_setting = inverted_residual_setting

        blocks = []
        self.drop_blocks = []

        input_channels = int(input_channels * width_mult)
        ori_blocks = model.blocks
        block_idx = 0
        for t, c, n, s, dr in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if self.drop_before_block[0]:
                    self.drop_blocks.append(DropClass(type=self.drop_before_block[1][0],
                                                      drop_config=self.drop_before_block[1][1]).cuda())
                    dr = 0
                stride = s if i == 0 else 1
                blocks.append(Q_LinearBottleneck(ori_blocks[block_idx], input_channels, output_channel, stride=stride,
                                                 expand_ratio=t, shift_bit=shift_bit, drop_config=dr, bias_bit=bias_bit,
                                                 **kwargs))
                input_channels = output_channel
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)
        self.drop_blocks = nn.Sequential(*self.drop_blocks)
        # change the final block
        self.quant_act_before_final_block = QuantAct(shift_bit=shift_bit)
        # self.sparseModel.add_module(str(i+1), QuantBnConv2d())
        # self.sparseModel[i+1].set_param(model.sparseModel[18], model.sparseModel[19])
        self.conv8 = QuantBnConv2d(per_channel=True, bias_bit=bias_bit, **kwargs)
        self.conv8.set_param(model.conv8, model.bn8)
        self.quant_act_int32_final = QuantAct(shift_bit=shift_bit)

        # in_channels = final_block_channels
        self.pool = QuantAveragePool2d()
        self.pool.set_param(model.pool)

        # self.sparseModel.add_module(str(i+2), QuantAveragePool2d(model.sparseModel[20].pool_size))
        # self.sparseModel.add_module(str(i+3), model.sparseModel[21])
        # self.sparseModel[i+2].set_param(model.sparseModel[20])
        self.quant_act_output = QuantAct(shift_bit=shift_bit)
        self.quant_act_int32 = QuantAct(shift_bit=shift_bit)

        self.fc = QuantLinear(bias_bit=bias_bit)
        self.fc.set_param(model.fc)

    def forward(self, x):
        # quantize input
        # x, act_scaling_factor = self.quant_act_before_first_block(x, None, None, None, None)
        act_scaling_factor = torch.ones(1).cuda()
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act_after_first_block(x, act_scaling_factor, weight_scaling_factor, None, None)

        # the init block
        # x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        # x = self.activatition_func(x)
        # x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

        # the feature block
        for i, channels in enumerate(self.blocks):
            cur_stage = getattr(self.blocks, str(i))
            if self.drop_before_block[0]:
                x, _ = self.drop_blocks[i](x)
            x, act_scaling_factor, _ = cur_stage((x, act_scaling_factor, None))

        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None)
        x, weight_scaling_factor = self.conv8(x, act_scaling_factor)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act_int32_final(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        # the final pooling
        x, act_scaling_factor = self.pool(x, act_scaling_factor)
        # x = self.sparseModel[i+3](x)

        # the output
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, None, None, None, None)
        x = SparseTensor(
            x.features.view(x.features.size(0), -1),
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        # x.features =
        x = self.fc(x, act_scaling_factor)
        if isinstance(x, torch.Tensor):
            return x
        else:
            return x.features


if __name__ == '__main__':
    from mink_mobilenetv2 import MobileNetV2ME

    # def denseToSparse(dense_tensor):
    #     non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
    #     locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)
    #     select_indices = non_zero_indices.split(1, dim=1)
    #     features = torch.squeeze(dense_tensor[select_indices], dim=-2)
    #     return locations, features

    rand_input = torch.rand(48, 159, 159, 2).cuda()
    model = MobileNetV2ME(in_channels=2, num_classes=10, width_mult=1)
    quant_model = Q_MobileNetV2(model, in_channels=2, num_classes=10, width_mult=1.0).cuda().train()
    _ = quant_model(denseToSparse(rand_input))
    a = 1