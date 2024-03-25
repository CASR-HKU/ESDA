"""
    Quantized MobileNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

import json
from .HAWQ_quant_module.quant_modules import *
import MinkowskiEngine as ME
from .utils import dense_to_sparse, RandomDrop
from MinkowskiEngine.MinkowskiSparseTensor import SparseTensor
from .mobilenet_submanifold import MobileNetSubmanifold


class MobileNetSubmanifoldQuant(nn.Module):
    def __init__(self, args):
        super(MobileNetSubmanifoldQuant, self).__init__()
        model_float32 = MobileNetSubmanifold(args)
        # self.model_float32 = model_float32
        if args.evaluate:
            self.model = Q_MobileNetV2(args, model_float32, in_channels=model_float32.sample_channel,
                                       num_classes=model_float32.num_classes)
            # self.model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        else:
            model_float32 = MobileNetSubmanifold(args)
            assert args.checkpoint, "Please specify the pretrained model if you want to quantize it."
            model_float32.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
            self.model = Q_MobileNetV2(args, model_float32, in_channels=model_float32.sample_channel,
                                       num_classes=model_float32.num_classes)
        self.use_heatmap = model_float32.use_heatmap

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)


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
        self.use_drop = False

    def forward(self, x, scaling_factor_int32=None, block_id=0, int_folder=""):
        if isinstance(x, tuple):
            x, scaling_factor_int32, block_id, int_folder = x
        if int_folder:
            np.save("{}/block_{}_input_scale.npy".format(int_folder, block_id),
                    scaling_factor_int32.cpu().numpy())
        if self.use_residual:
            identity = x

        # x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None)
        act_scaling_factor = scaling_factor_int32
        identity_scaling_factor = act_scaling_factor

        # if self.use_exp_conv:
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor, "block_{}_conv1".format(block_id), int_folder)
        x = self.activation_func(x)

        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None, 
                                                name="block_{}_conv1".format(block_id), int_folder=int_folder)

        if int_folder:
            np.save("{}/block_{}_conv1_act_out.npy".format(int_folder, block_id),
                    act_scaling_factor.cpu().numpy())

        x, weight_scaling_factor = self.conv2(x, act_scaling_factor, "block_{}_conv2".format(block_id), int_folder)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None,
                                                name="block_{}_conv2".format(block_id), int_folder=int_folder)

        # note that, there is no activation for the last conv
        x, weight_scaling_factor = self.conv3(x, act_scaling_factor, "block_{}_conv3".format(block_id), int_folder)

        if self.use_residual:
            x = x + identity
            if int_folder:
                np.save("{}/block_{}_identity_inp_int.npy".format(int_folder, block_id),
                        (identity.F/identity_scaling_factor).cpu().numpy())
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_scaling_factor, None, 
                                                         name="block_{}_conv3".format(block_id), int_folder=int_folder)
            if int_folder:
                np.save("{}/block_{}_identity_added_out_integer.npy".format(int_folder, block_id),
                        (x.F/act_scaling_factor).cpu().numpy())
                np.save("{}/block_{}_identity_added_out_out_C_stride{}.npy".
                        format(int_folder, block_id, x.tensor_stride[0]), x.C.cpu().numpy())
        else:
            x, act_scaling_factor = self.quant_act_int32(
                x, act_scaling_factor, weight_scaling_factor, None, None, None, name="block_{}_conv3".format(block_id),
                int_folder=int_folder)
        
        if int_folder:
            np.save("{}/block_{}_conv3_act_out.npy".format(int_folder, block_id), act_scaling_factor.cpu().numpy())
            np.save("{}/block_{}_conv3_output_coordinate_stride{}.npy".format(int_folder, block_id, x.tensor_stride[0]),
                    x.C.cpu().numpy())
            np.save("{}/block_{}_conv3_output_integer.npy".format(int_folder, block_id),
                    (x.F / act_scaling_factor).cpu().numpy())
        return x, act_scaling_factor




class Q_MobileNetV2(nn.Module):
    def __init__(self,
                 args,
                 model,
                 in_channels,
                 num_classes=1000,
                 **kwargs):
        super(Q_MobileNetV2, self).__init__()

        self.device = args.device
        self.bias_bit = args.bias_bit
        self.shift_bit = args.shift_bit
        self.conv1_bit = args.conv1_bit

        with open(args.model_cfg, 'r') as ft:
            cfg = json.load(ft)

        self.width_mult = 1.0
        if "width_mult" in cfg:
            self.width_mult = cfg["width_mult"]
        self.input_channel = cfg["input_channel"]
        self.last_channel = cfg["last_channel"]
        self.inverted_residual_setting = cfg["backbone"]

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.use_heatmap = False

        self.quant_act_before_first_block = QuantAct(shift_bit=self.shift_bit)
        self.conv1 = QuantBnConv2d(per_channel=True, weight_bit=self.conv1_bit, bias_bit=self.bias_bit, **kwargs)
        self.conv1.set_param(model.features[0].conv, model.features[0].norm)
        self.quant_act_after_first_block = QuantAct(shift_bit=self.shift_bit)

        self.activation_func = ME.MinkowskiReLU6(inplace=True)

        blocks = []

        input_channels = int(self.input_channel * self.width_mult)
        ori_blocks = model.features[1:-1]
        block_idx = 0
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(Q_LinearBottleneck(ori_blocks[block_idx], input_channels, output_channel, stride=stride,
                                                 expand_ratio=t, shift_bit=self.shift_bit, bias_bit=self.bias_bit,
                                                 **kwargs))
                input_channels = output_channel
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)
        # self.drop_blocks = nn.Sequential(*self.drop_blocks)
        # change the final block
        self.quant_act_before_final_block = QuantAct(shift_bit=self.shift_bit)
        # self.sparseModel.add_module(str(i+1), QuantBnConv2d())
        # self.sparseModel[i+1].set_param(model.sparseModel[18], model.sparseModel[19])
        self.conv8 = QuantBnConv2d(per_channel=True, bias_bit=self.bias_bit, **kwargs)
        self.conv8.set_param(model.features[-1].conv, model.features[-1].norm)
        self.quant_act_int32_final = QuantAct(shift_bit=self.shift_bit)

        # in_channels = final_block_channels
        self.pool = QuantAveragePool2d()
        self.pool.set_param(model.pool)

        self.quant_act_output = QuantAct(shift_bit=self.shift_bit)
        self.quant_act_int32 = QuantAct(shift_bit=self.shift_bit)

        if "heatmap" in cfg and cfg["heatmap"]:
            self.use_heatmap = True
            raise NotImplementedError
        else:
            self.use_heatmap = False
            self.rnn = model.rnn
            self.fc = model.classifier[0]


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

    def forward(self, x, int_folder=""):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(batch_size*seq_len, channels, height, width)
        x = x.permute(0, 2, 3, 1)

        coord, feat = dense_to_sparse(x)
        x = SparseTensor(features=feat.contiguous(), coordinates=coord.int().contiguous(), device=self.device)

        # quantize input
        x, act_scaling_factor = self.quant_act_before_first_block(x, None, None, None, None,
                                                                  name="input", int_folder=int_folder)
        # act_scaling_factor = torch.ones(1).cuda()
        # x, act_scaling_factor = self.quant_input(x, act_scaling_factor, weight_scaling_factor, None, None,
        #                                          name="input", int_folder=int_folder)
        x, weight_scaling_factor = self.conv1(x, act_scaling_factor, "conv1", int_folder)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act_after_first_block(x, act_scaling_factor, weight_scaling_factor, None, None, name="conv1", int_folder=int_folder)

        if int_folder:
            np.save("{}/conv1_act_out.npy".format(int_folder), act_scaling_factor.cpu().numpy())
            np.save("{}/conv1_output_coordinate_stride{}.npy".format(int_folder, x.tensor_stride[0]), x.C.cpu().numpy())
            np.save("{}/conv1_output_integer.npy".format(int_folder),
                    np.round((x.F / act_scaling_factor).cpu().numpy()))

        # the feature block
        for i, channels in enumerate(self.blocks):
            cur_stage = getattr(self.blocks, str(i))
            x, act_scaling_factor = cur_stage((x, act_scaling_factor, i, int_folder))

        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None)
        x, weight_scaling_factor = self.conv8(x, act_scaling_factor, "conv8", int_folder)
        x = self.activation_func(x)
        x, act_scaling_factor = self.quant_act_int32_final(x, act_scaling_factor, weight_scaling_factor, None, None, None, name="conv8", int_folder=int_folder)

        if int_folder:
            np.save("{}/conv8_act_out.npy".format(int_folder), act_scaling_factor.cpu().numpy())
            np.save("{}/conv8_output_coordinate_stride{}.npy".format(int_folder, x.tensor_stride[0]), x.C.cpu().numpy())
            np.save("{}/conv8_output_integer.npy".format(int_folder), (x.F / act_scaling_factor).cpu().numpy())

        # the final pooling
        x, act_scaling_factor = self.pool(x, act_scaling_factor)

        # the output
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, None, None, None, None)
        # x = SparseTensor(
        #     x.features.view(x.features.size(0), -1),
        #     coordinate_map_key=x.coordinate_map_key,
        #     coordinate_manager=x.coordinate_manager,
        # )

        x = x.F.view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        # x = self.fc(x, act_scaling_factor)
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