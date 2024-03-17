import json
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import os
import math
import argparse
import MinkowskiEngine as ME
from torch.nn import Parameter
import torch.nn as nn

from common import *


def sort_coo_coordinate(coords):
    max_y = torch.max(coords[:, 1])
    max_x = torch.max(coords[:, 2])
    key = coords[:, 0] * max_x * max_y + coords[:, 1] * max_x + coords[:, 2]
    sorted_key, sorted_idx = torch.sort(key)
    return sorted_idx


def pack_3x3_dw_weights(weights, PF, precision=8):
    ## assuming weight in [9][IC] format
    # change weight into int
    # assume precision is 8 currently

    assert precision == 8

    weights = weights.astype(int)
    C = weights.shape[1]
    assert C % PF == 0, f"weights.shape: {weights.shape}, C: {C}, PF: {PF}"
    output_array = np.zeros((9, C // PF), dtype=object)
    for k in range(9):
        for c in range(C // PF):
            tmp = 0
            for i in range(PF):
                tmp += int(weights[k][c * PF + i] & 0xFF) << (precision * i)
            if tmp >= (1 << (8 * PF - 1)):
                tmp -= 1 << (8 * PF)

            output_array[k][c] = tmp

    return output_array


def pack_3x3_weights(weights, PF, precision=8):
    ## assuming weight in [9][OC][IC] format
    # change weight into int
    # assume precision is 8 currently

    assert precision == 8

    weights = weights.astype(int)
    _, OC, IC = weights.shape

    # print("weights.shape", weights.shape, "OC:", OC, " IC:", IC, "PF:", PF)
    output_array = np.zeros((9, OC), dtype=object)
    for k in range(9):
        for oc in range(OC):
            tmp = 0
            for ic in range(IC):
                tmp += int(weights[k][oc][ic] & 0xFF) << (precision * ic)
            if tmp >= (1 << (8 * PF - 1)):
                tmp -= 1 << (8 * PF)
            output_array[k][oc] = tmp

    return output_array


def pack_1x1_weights(weights, PI, precision=8):
    assert precision == 8
    weights = weights.astype(int)
    OC, IC = weights.shape
    assert IC % PI == 0, f"weights.shape:{weights.shape}, IC: {IC}, PI: {PI}"
    output_array = np.zeros((OC, IC // PI), dtype=object)
    for oc in range(OC):
        for ic in range(IC // PI):
            tmp = 0
            for i in range(PI):
                w = int(weights[oc][ic * PI + i] & 0xFF)
                tmp += w << (precision * i)
            if tmp >= (1 << (8 * PI - 1)):
                tmp -= 1 << (8 * PI)
            output_array[oc][ic] = tmp
    return output_array


def pack_scale_bias(scale, bias, W):
    if scale.shape != bias.shape:
        raise ValueError("Arrays must have the same shape.")

    output_array = np.zeros(scale.shape, dtype=object)
    scale = scale.astype(np.int64)
    bias = bias.astype(np.int64)
    for i in range(scale.shape[0]):
        output_array[i] = (bias[i] & (2**W - 1)) + ((scale[i] & (2**W - 1)) << W)
        if output_array[i] >= (1 << (2 * W - 1)):
            output_array[i] -= 1 << (2 * W)
        # print(i, " scale:", scale[i], " bias:", bias[i], output_array[i])
    return output_array


def save_array_as_c_init(array, output_file):
    with open(output_file, "w") as f:
        write_array_elements(array, f)


def write_array_elements(array, file):
    if array.ndim == 1:
        for i, element in enumerate(array):
            file.write(f'"{str(element)}"')
            if i < len(array) - 1:
                file.write(", ")
    else:
        for i in range(array.shape[0]):
            file.write("{")
            write_array_elements(array[i], file)
            file.write("},\n")


class tb_module_1x1(nn.Module):
    def __init__(
        self,
        int_weight,
        int_bias,
        int_scale,
        PI,
        shift_n=8,
        kernel_size=1,
        out_dir="int",
        prefix="",
        relu=True,
        bias_bits=32,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.prefix = prefix
        self.batch = 1
        self.IC = int_weight.shape[0]
        self.OC = int_weight.shape[1]
        self.shift_n = shift_n
        self.bias_bits = bias_bits
        self.PI = PI
        self.relu = relu
        self.conv_kernel = ME.MinkowskiConvolution(
            in_channels=self.IC,
            out_channels=self.OC,
            kernel_size=kernel_size,
            stride=1,
            dimension=2,
        )
        self.weights = torch.round(torch.from_numpy(int_weight)).float()
        self.conv_kernel.kernel = Parameter(self.weights)
        self.scale = torch.round(torch.from_numpy(int_scale))
        self.bias = torch.round(torch.from_numpy(int_bias))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def dump_weights(self):
        weights = self.weights.clone().detach()
        weights = weights.transpose(0, 1)

        packed_weights = pack_1x1_weights(weights.numpy(), self.PI, precision=8)

        weights = np.ndarray.flatten(weights.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_weight.txt"), weights, fmt="%d"
        )
        scale = self.scale.clone().detach()
        scale = np.ndarray.flatten(scale.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_scale.txt"), scale, fmt="%d"
        )
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_scale_c.txt"),
            scale,
            fmt="%d",
            newline=",",
        )

        bias = self.bias.clone().detach()
        bias = np.ndarray.flatten(bias.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_bias.txt"), bias, fmt="%d"
        )
        # print("scale:", scale)
        # print("bias:", bias)
        scale_bias = pack_scale_bias(scale, bias, self.bias_bits)
        save_array_as_c_init(
            scale_bias, os.path.join(self.out_dir, self.prefix + "_s.txt")
        )
        save_array_as_c_init(
            packed_weights,
            os.path.join(self.out_dir, self.prefix + "_w.txt"),
        )

    def dump_results(self):
        sorted_idx = sort_coo_coordinate(self.psum_coords)
        psum = self.psum.clone().detach()
        psum = psum[sorted_idx]
        psum = np.ndarray.flatten(psum.numpy())
        layer_output = self.layer_output.clone().detach()
        layer_output = layer_output[sorted_idx]
        layer_output = np.ndarray.flatten(layer_output.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_psum.txt"), psum, fmt="%d"
        )
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_output.txt"),
            layer_output,
            fmt="%d",
        )

    def sort_coo_coordinate(self, coords):
        max_y = torch.max(coords[:, 1])
        max_x = torch.max(coords[:, 2])
        key = coords[:, 0] * max_x * max_y + coords[:, 1] * max_x + coords[:, 2]
        sorted_key, sorted_idx = torch.sort(key)
        return sorted_idx

    def forward(self, int_inp):
        psum_sparse_tensor = self.conv_kernel(int_inp)  # .expand_as(self.psum)
        self.psum, self.psum_coords, tensor_stride = (
            psum_sparse_tensor.F,
            psum_sparse_tensor.C,
            psum_sparse_tensor.tensor_stride,
        )

        add_biased_sparse_tensor = (self.bias).expand_as(
            self.psum
        ) + psum_sparse_tensor.F

        mul = add_biased_sparse_tensor * self.scale.expand_as(add_biased_sparse_tensor)
        round_shift = pow(2, self.shift_n - 1)
        mul += round_shift
        mul = mul.type(torch.int64)
        mul_flatten = mul[sort_coo_coordinate(self.psum_coords)].numpy().flatten()
        self.psum_mult_shift = (mul) >> self.shift_n

        self.psum_mult_shift = self.psum_mult_shift.type(torch.FloatTensor)
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -128, 127)
        if self.relu:
            self.psum_mult_shift_clamp = F.relu(self.psum_mult_shift_clamp)
        self.layer_output = self.psum_mult_shift_clamp
        output_sparse_tensor = ME.SparseTensor(
            features=self.psum_mult_shift_clamp,
            coordinates=self.psum_coords,
            tensor_stride=tensor_stride,
        )
        return output_sparse_tensor


class tb_module_1x1_residual(nn.Module):
    def __init__(
        self,
        int_weight,
        int_bias,
        int_scale,
        int_idscale,
        PI,
        shift_n=8,
        kernel_size=1,
        out_dir="int",
        prefix="",
        bias_bits=32,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.prefix = prefix
        self.batch = 1
        self.IC = int_weight.shape[0]
        self.OC = int_weight.shape[1]
        self.shift_n = shift_n
        self.bias_bits = bias_bits
        self.PI = PI
        self.conv_kernel = ME.MinkowskiConvolution(
            in_channels=self.IC,
            out_channels=self.OC,
            kernel_size=kernel_size,
            stride=1,
            dimension=2,
        )

        self.weights = torch.round(torch.from_numpy(int_weight)).float()
        self.conv_kernel.kernel = Parameter(self.weights)
        self.scale = torch.round(torch.from_numpy(int_scale))
        self.bias = torch.round(torch.from_numpy(int_bias))
        self.id_scale = torch.round(torch.from_numpy(int_idscale))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def dump_weights(self):
        weights = self.weights.clone().detach()
        weights = weights.transpose(0, 1)
        packed_weights = pack_1x1_weights(weights.numpy(), self.PI)

        weights = np.ndarray.flatten(weights.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_weight.txt"), weights, fmt="%d"
        )
        scale = self.scale.clone().detach()
        scale = np.ndarray.flatten(scale.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_scale.txt"), scale, fmt="%d"
        )
        bias = self.bias.clone().detach()
        bias = np.ndarray.flatten(bias.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_bias.txt"), bias, fmt="%d"
        )
        scale_bias = pack_scale_bias(scale, bias, self.bias_bits)
        save_array_as_c_init(
            scale_bias, os.path.join(self.out_dir, self.prefix + "_s.txt")
        )
        save_array_as_c_init(
            packed_weights,
            os.path.join(self.out_dir, self.prefix + "_w.txt"),
        )

    def dump_results(self):
        sorted_idx = sort_coo_coordinate(self.psum_coords)
        psum = self.psum.clone().detach()
        psum = psum[sorted_idx]
        psum = np.ndarray.flatten(psum.numpy())
        layer_output = self.layer_output.clone().detach()
        layer_output = layer_output[sorted_idx]
        layer_output = np.ndarray.flatten(layer_output.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_psum.txt"), psum, fmt="%d"
        )
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_output.txt"),
            layer_output,
            fmt="%d",
        )

    def forward(self, int_inp, conv1_int_input):
        psum_sparse_tensor = self.conv_kernel(int_inp).detach()  # .expand_as(self.psum)
        self.psum, self.psum_coords, tensor_stride = (
            psum_sparse_tensor.F,
            psum_sparse_tensor.C,
            psum_sparse_tensor.tensor_stride,
        )

        add_biased_sparse_tensor = self.bias.expand_as(self.psum) + psum_sparse_tensor.F

        mul = add_biased_sparse_tensor * self.scale.expand_as(add_biased_sparse_tensor)
        round_shift = pow(2, self.shift_n - 1)
        mul = mul.type(torch.int64)
        mul += round_shift

        # print("mul:", mul[sort_coo_coordinate(self.psum_coords)])
        mul = (mul) >> self.shift_n

        int_identity = torch.round(self.id_scale) * conv1_int_input
        int_identity = int_identity.type(torch.int64)
        int_identity += round_shift

        int_identity = (int_identity) >> self.shift_n

        id_flatten = (
            int_identity[sort_coo_coordinate(self.psum_coords)].numpy().flatten()
        )
        # print("int_identity:", int_identity[sort_coo_coordinate(self.psum_coords)])
        mul_flatten = mul[sort_coo_coordinate(self.psum_coords)].numpy().flatten()

        self.psum_mult_shift = mul + int_identity

        self.psum_mult_shift = self.psum_mult_shift.type(torch.FloatTensor)
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -128, 127)
        self.layer_output = self.psum_mult_shift_clamp
        output_sparse_tensor = ME.SparseTensor(
            features=self.psum_mult_shift_clamp,
            coordinates=self.psum_coords,
            tensor_stride=tensor_stride,
        )
        return output_sparse_tensor


class tb_module_3x3(nn.Module):
    def __init__(
        self,
        int_weight,
        int_bias,
        int_scale,
        shift_n=8,
        stride=1,
        out_dir="int",
        prefix="",
        bias_bits=32,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.prefix = prefix
        self.batch = 1
        self.IC = int_weight.shape[1]
        self.OC = int_weight.shape[2]
        self.shift_n = shift_n
        self.bias_bits = bias_bits
        self.stride = stride
        self.PI = self.IC
        self.conv_kernel = ME.MinkowskiConvolution(
            in_channels=self.IC,
            out_channels=self.OC,
            kernel_size=3,
            stride=self.stride,
            dimension=2,
        )
        # print(self.conv_kernel.kernel.shape)
        self.weights = torch.round(torch.from_numpy(int_weight)).float()
        self.conv_kernel.kernel = Parameter(self.weights)
        self.scale = torch.round(torch.from_numpy(int_scale))
        self.bias = torch.round(torch.from_numpy(int_bias))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def dump_weights(self):
        weights = self.weights.clone().detach()
        weights = torch.reshape(weights, (3, 3, self.IC, self.OC))
        weights = weights.transpose(0, 1)
        weights_to_pack = torch.reshape(weights, (9, self.IC, self.OC))
        weights_to_pack = weights_to_pack.transpose(1, 2)
        packed_weights = pack_3x3_weights(
            weights_to_pack.numpy(), PF=self.PI, precision=8
        )
        weights = np.ndarray.flatten(weights.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_weight.txt"), weights, fmt="%d"
        )
        scale = self.scale.clone().detach()
        scale = np.ndarray.flatten(scale.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_scale.txt"), scale, fmt="%d"
        )
        bias = self.bias.clone().detach()
        bias = np.ndarray.flatten(bias.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_bias.txt"), bias, fmt="%d"
        )
        scale_bias = pack_scale_bias(scale, bias, self.bias_bits)
        save_array_as_c_init(
            scale_bias, os.path.join(self.out_dir, self.prefix + "_s.txt")
        )
        save_array_as_c_init(
            packed_weights,
            os.path.join(self.out_dir, self.prefix + "_w.txt"),
        )

    def dump_results(self):
        sorted_idx = sort_coo_coordinate(self.psum_coords)
        psum = self.psum.clone().detach()
        psum = psum[sorted_idx]
        psum = np.ndarray.flatten(psum.numpy())
        layer_output = self.layer_output.clone().detach()
        layer_output = layer_output[sorted_idx]
        layer_output = np.ndarray.flatten(layer_output.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_psum.txt"), psum, fmt="%d"
        )
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_output.txt"),
            layer_output,
            fmt="%d",
        )

    def forward(self, int_inp):
        psum_sparse_tensor = self.conv_kernel(int_inp)
        self.psum, self.psum_coords, tensor_stride = (
            psum_sparse_tensor.F,
            psum_sparse_tensor.C,
            psum_sparse_tensor.tensor_stride,
        )

        add_biased_sparse_tensor = self.bias.expand_as(self.psum) + psum_sparse_tensor.F

        mul = add_biased_sparse_tensor * self.scale.expand_as(add_biased_sparse_tensor)
        round_shift = pow(2, self.shift_n - 1)
        mul += round_shift
        mul = mul.type(torch.int64)

        self.psum_mult_shift = (mul) >> self.shift_n
        self.psum_mult_shift = self.psum_mult_shift.type(torch.FloatTensor)
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -128, 127)
        self.psum_mult_shift_clamp_relu = F.relu(self.psum_mult_shift_clamp)
        self.layer_output = self.psum_mult_shift_clamp_relu
        output_sparse_tensor = ME.SparseTensor(
            features=self.psum_mult_shift_clamp_relu,
            coordinates=self.psum_coords,
            tensor_stride=tensor_stride,
        )
        return output_sparse_tensor


class tb_module_3x3_dw(nn.Module):
    def __init__(
        self,
        int_weight,
        int_bias,
        int_scale,
        PI,
        shift_n=8,
        stride=1,
        out_dir="int",
        prefix="",
        bias_bits=32,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.prefix = prefix
        self.batch = 1
        self.IC = int_weight.shape[1]
        self.shift_n = shift_n
        self.bias_bits = bias_bits
        self.stride = stride
        self.PI = PI
        self.conv_kernel = ME.MinkowskiChannelwiseConvolution(
            in_channels=self.IC, kernel_size=3, stride=self.stride, dimension=2
        )
        self.weights = torch.round(torch.from_numpy(int_weight)).float()
        self.conv_kernel.kernel = Parameter(self.weights)
        self.scale = torch.round(torch.from_numpy(int_scale))
        self.bias = torch.round(torch.from_numpy(int_bias))
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def dump_weights(self):
        weights = self.weights.clone().detach()
        weights = torch.reshape(weights, (3, 3, self.IC))
        weights = weights.transpose(0, 1)
        weights_to_pack = torch.reshape(weights, (9, self.IC))
        packed_weights = pack_3x3_dw_weights(
            weights_to_pack.numpy(), PF=self.PI, precision=8
        )
        weights = np.ndarray.flatten(weights.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_weight.txt"), weights, fmt="%d"
        )
        scale = self.scale.clone().detach()
        scale = np.ndarray.flatten(scale.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_scale.txt"), scale, fmt="%d"
        )
        bias = self.bias.clone().detach()
        bias = np.ndarray.flatten(bias.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_bias.txt"), bias, fmt="%d"
        )
        scale_bias = pack_scale_bias(scale, bias, self.bias_bits)
        save_array_as_c_init(
            scale_bias, os.path.join(self.out_dir, self.prefix + "_s.txt")
        )
        save_array_as_c_init(
            packed_weights,
            os.path.join(self.out_dir, self.prefix + "_w.txt"),
        )

    def dump_results(self):
        sorted_idx = sort_coo_coordinate(self.psum_coords)
        psum = self.psum.clone().detach()
        psum = psum[sorted_idx]
        psum = np.ndarray.flatten(psum.numpy())
        layer_output = self.layer_output.clone().detach()
        layer_output = layer_output[sorted_idx]
        layer_output = np.ndarray.flatten(layer_output.numpy())
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_psum.txt"), psum, fmt="%d"
        )
        np.savetxt(
            os.path.join(self.out_dir, self.prefix + "_output.txt"),
            layer_output,
            fmt="%d",
        )

    def forward(self, int_inp):
        psum_sparse_tensor = self.conv_kernel(int_inp)
        self.psum, self.psum_coords, tensor_stride = (
            psum_sparse_tensor.F,
            psum_sparse_tensor.C,
            psum_sparse_tensor.tensor_stride,
        )

        add_biased_sparse_tensor = self.bias.expand_as(self.psum) + psum_sparse_tensor.F

        mul = add_biased_sparse_tensor * self.scale.expand_as(add_biased_sparse_tensor)
        round_shift = pow(2, self.shift_n - 1)
        mul += round_shift
        mul = mul.type(torch.int64)

        self.psum_mult_shift = (mul) >> self.shift_n
        self.psum_mult_shift = self.psum_mult_shift.type(torch.FloatTensor)
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -128, 127)
        self.psum_mult_shift_clamp_relu = F.relu(self.psum_mult_shift_clamp)
        self.layer_output = self.psum_mult_shift_clamp_relu
        output_sparse_tensor = ME.SparseTensor(
            features=self.psum_mult_shift_clamp_relu,
            coordinates=self.psum_coords,
            tensor_stride=tensor_stride,
        )
        return output_sparse_tensor


class tb_1x1_3x3dw_1x1_block(nn.Module):
    def __init__(
        self,
        out_dir,
        name,
        stride,
        use_residual,
        shift_n,
        weight_0,
        bias_0,
        scale_0,
        weight_1,
        bias_1,
        scale_1,
        weight_2,
        bias_2,
        scale_2,
        id_scale,
        PI_0,
        PI_1,
        PI_2,
        bias_bit,
    ):
        super().__init__()
        self.out_dir = out_dir
        self.name = name
        self.batch = 1
        self.stride = stride
        self.shift_n = shift_n
        self.use_residual = use_residual
        self.id_scale = id_scale
        self.module_1x1 = tb_module_1x1(
            weight_0,
            bias_0,
            scale_0,
            PI=PI_0,
            out_dir=self.out_dir,
            prefix=f"{name}_0",
            shift_n=self.shift_n,
            bias_bits=bias_bit,
        )
        self.module_3x3_dw = tb_module_3x3_dw(
            weight_1,
            bias_1,
            scale_1,
            PI=PI_1,
            out_dir=self.out_dir,
            prefix=f"{name}_1",
            stride=self.stride,
            shift_n=self.shift_n,
            bias_bits=bias_bit,
        )
        if self.use_residual:
            self.module_1x1_2 = tb_module_1x1_residual(
                weight_2,
                bias_2,
                scale_2,
                id_scale,
                PI=PI_1,
                out_dir=self.out_dir,
                prefix=f"{name}_2",
                shift_n=self.shift_n,
                bias_bits=bias_bit,
            )
        else:
            self.module_1x1_2 = tb_module_1x1(
                weight_2,
                bias_2,
                scale_2,
                PI=PI_1,
                out_dir=self.out_dir,
                prefix=f"{name}_2",
                shift_n=self.shift_n,
                relu=False,
                bias_bits=bias_bit,
            )

    def dump_weights(self):
        self.module_1x1.dump_weights()
        self.module_3x3_dw.dump_weights()
        self.module_1x1_2.dump_weights()
        if self.use_residual:
            np.savetxt(
                os.path.join(self.out_dir, f"{self.name}_id_scale.txt"),
                np.ndarray.flatten(self.id_scale),
                fmt="%d",
            )
            save_array_as_c_init(
                self.id_scale.astype(np.int64).flatten(),
                os.path.join(self.out_dir, f"{self.name}_i.txt"),
            )

    def dump_results(self):
        self.module_1x1.dump_results()
        self.module_3x3_dw.dump_results()
        self.module_1x1_2.dump_results()

    def forward(self, input_sparse_tensor):
        identity = input_sparse_tensor.F
        output_sparse_tensor = self.module_1x1(input_sparse_tensor)
        output_sparse_tensor = self.module_3x3_dw(output_sparse_tensor)
        if self.use_residual:
            output_sparse_tensor = self.module_1x1_2(output_sparse_tensor, identity)
        else:
            output_sparse_tensor = self.module_1x1_2(output_sparse_tensor)
        return output_sparse_tensor


def dense_to_sparse(dense):
    non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense[select_indices], dim=-2)
    return non_zero_indices, features


def generate_mask(coordinate, tensor_stride, input_height, input_width):
    coordinate = np.ceil(coordinate / tensor_stride).astype(np.int32)
    mask = np.zeros(
        (
            math.ceil(input_height / tensor_stride),
            math.ceil(input_width / tensor_stride),
        ),
        dtype=np.uint8,
    )
    # print("mask.shape", mask.shape)
    # print("coordinate.max", coordinate.max(axis=0))
    mask[coordinate[:, 1], coordinate[:, 2]] = 1
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", "-d", type=str, default=".")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--save_dir", '-s', type=str, default="")
    parser.add_argument("--shift_bit", type=int, default=32)
    parser.add_argument("--out_dir", "-o", type=str, default="")

    args = parser.parse_args()

    # load cfg
    if not os.path.exists(os.path.join(args.work_dir, "cfg.json")):
        raise FileNotFoundError("cfg.json not found.")
    cfg = json.load(open(os.path.join(args.work_dir, "cfg.json")))
    # check out_dir
    out_dir = os.path.join(args.work_dir, "data") if args.out_dir == "" else args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # check data_dir
    data_dir = cfg["model_path"]
    if args.data_dir:
        data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir: {data_dir} not found.")

    fpath_F = os.path.join(data_dir, 'conv1_input_integer.npy')
    input_F = np.load(fpath_F)
    fpath_C = os.path.join(data_dir, 'conv1_input_coordinate_stride1.npy')
    input_C = np.load(fpath_C)
    if args.save_dir:
        dpath_F = os.path.join(args.save_dir, 'conv1_input_integer.npy')
        dpath_C = os.path.join(args.save_dir, 'conv1_input_coordinate_stride1.npy')
        raw_json = os.path.join(data_dir, 'model.json')
        dest_json = os.path.join(args.save_dir, 'model.json')
        shutil.copy(fpath_F, dpath_F)
        shutil.copy(fpath_C, dpath_C)
        shutil.copy(raw_json, dest_json)

    # parse cfg
    # dataset = cfg["dataset"]
    if cfg["dataset"] == "NCAL":
        shift_n = 32
        bias_bit = 32
    else:
        shift_n = 16
        bias_bit = 16

    input_h, input_w = cfg["input_shape"]
    # for each layer
    for layer_i, layer in enumerate(cfg["layers"]):
        name = legal_low(layer["name"])
        channels = layer["channels"]
        print(f"Processing layer {name}...")
        print("channels", channels)
        parallelism = layer["parallelism"]
        residual = layer["residual"]
        stride = layer["stride"]
        in_tensor_stride = layer["tensor_stride"][0]
        out_tensor_stride = layer["tensor_stride"][-1]
        if layer["type"] == "conv":
            module_kwargs = {}
            for dtype, fname in zip(
                ["weight", "bias", "scale"],
                ["weight_integer", "bias_integer", "Sw_bar"],
            ):
                fpath = os.path.join(data_dir, npy_of(layer, "", fname))
                if args.save_dir:
                    dpath = os.path.join(args.save_dir, npy_of(layer, "", fname))
                    shutil.copy(fpath, dpath)
                module_kwargs[f"int_{dtype}"] = np.load(fpath)
                print(
                    f"int_{dtype}.shape",
                    module_kwargs[f"int_{dtype}"].shape,
                    module_kwargs[f"int_{dtype}"].size,
                )
            if layer["name"] == "conv1":
                assert (
                    module_kwargs[f"int_weight"].size == 9 * channels[0] * channels[1]
                )
            else:
                assert module_kwargs[f"int_weight"].size == channels[0] * channels[1]
            assert module_kwargs[f"int_bias"].size == channels[1]
            assert module_kwargs[f"int_scale"].size == channels[1]
            if layer["name"] == "conv8":
                module_kwargs["PI"] = parallelism[0]
            module_kwargs["shift_n"] = shift_n
            if layer["name"] == "conv8":
                module_kwargs["kernel_size"] = 1
            if layer["name"] == "conv1":
                module_kwargs["stride"] = stride
            module_kwargs["out_dir"] = out_dir
            module_kwargs["prefix"] = name
            if layer["name"] == "conv8":
                module_kwargs["relu"] = True
            module_kwargs["bias_bits"] = bias_bit
            if layer["name"] == "conv1":
                test_module = tb_module_3x3(**module_kwargs)
            elif layer["name"] == "conv8":
                test_module = tb_module_1x1(**module_kwargs)
            else:
                raise ValueError(f"Unknown layer name: {layer['name']}")
        elif layer["type"] == "block":
            module_kwargs = {}
            for i in range(3):
                for dtype, fname in zip(
                    ["weight", "bias", "scale"],
                    ["weight_integer", "bias_integer", "Sw_bar"],
                ):
                    k = f"{dtype}_{i}"
                    fpath = os.path.join(data_dir, npy_of(layer, i + 1, fname))
                    if args.save_dir:
                        dpath = os.path.join(args.save_dir, npy_of(layer, i + 1, fname))
                        shutil.copy(fpath, dpath)
                    module_kwargs[k] = np.load(fpath)
                    print(f"{k}.shape", module_kwargs[k].shape, module_kwargs[k].size)
                tmp_ch = [
                    [channels[0], channels[1]],
                    [9, channels[1]],
                    [channels[1], channels[2]],
                ][i]
                print(f"tmp_ch: {tmp_ch}")
                assert module_kwargs[f"weight_{i}"].size == tmp_ch[0] * tmp_ch[1]
                assert module_kwargs[f"bias_{i}"].size == tmp_ch[1]
                assert module_kwargs[f"scale_{i}"].size == tmp_ch[1]
            if residual:
                fpath = os.path.join(data_dir, npy_of(layer, 3, "Sid_bar"))
                module_kwargs["id_scale"] = np.load(fpath)
            else:
                module_kwargs["id_scale"] = None
            module_kwargs["shift_n"] = shift_n
            module_kwargs["out_dir"] = out_dir
            module_kwargs["name"] = name
            module_kwargs["use_residual"] = residual
            module_kwargs["stride"] = stride
            module_kwargs["PI_0"] = parallelism[0]
            module_kwargs["PI_1"] = parallelism[1]
            module_kwargs["PI_2"] = parallelism[2]
            module_kwargs["bias_bit"] = bias_bit
            # build module
            test_module = tb_1x1_3x3dw_1x1_block(**module_kwargs)
        elif layer["type"] == "linear":
            pooled_feat = output_st.F.sum(dim=0, keepdim=True)
            with open(os.path.join(out_dir, "tb_output.txt"), "w") as f:
                for i in pooled_feat.tolist()[0]:
                    f.write(str(int(i)) + "\n")
            continue

            # input_tensor = output_st.F
            # pooled_feat = input_tensor.sum(dim=0, keepdim=True)
            # fpath = os.path.join(data_dir, npy_of(layer, "", "weight_integer"))
            if args.save_dir:
                dpath = os.path.join(args.save_dir, npy_of(layer, "", "weight_integer"))
                shutil.copy(fpath, dpath)
            w_int = np.load(fpath)
            out = np.matmul(pooled_feat, np.transpose(w_int))
            pred = np.argmax(out, axis=1)
            out_gt = np.load(os.path.join(data_dir, "output_logit.npy"))
            pred_gt = np.argmax(out_gt, axis=1)

            print("w_int.shape", w_int.shape, w_int.size)
            packed_weights = pack_1x1_weights(
                w_int, PI=layer["parallelism"][0], precision=8
            )
            save_array_as_c_init(
                packed_weights, os.path.join(out_dir, f"{layer['name']}_w.txt")
            )

            print("out", out)
            print("out_gt", out_gt)
            print("pred:", pred)
            print("pred_gt:", pred_gt)
            with open(os.path.join(out_dir, "tb_output.txt"), "w") as f:
                for i in out.tolist()[0]:
                    f.write(str(int(i)) + "\n")
            continue
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
        # load input
        # fpath = os.path.join(data_dir, npy_of(layer, 1, "input_integer"))
        # input_F = np.load(fpath)
        # fpath = os.path.join(
        #     data_dir, npy_of(layer, 1, f"input_coordinate_stride{in_tensor_stride}")
        # )
        # input_C = np.load(fpath)
        if layer_i == 0:
            input_st = ME.SparseTensor(
                features=torch.from_numpy(input_F),
                coordinates=torch.from_numpy(input_C),
                tensor_stride=in_tensor_stride,
            )
        else:
            input_st = output_st
            input_C = input_st.C.numpy()
            input_F = input_st.F
            first_sample_idx = max([i for i, item in enumerate(input_C) if item[0] == 0])
            input_C = input_C[:first_sample_idx]
            input_F = input_F[:first_sample_idx]
        # generate mask
        # mask = generate_mask(input_C, in_tensor_stride, input_h, input_w)
        # build input sparse tensor
        # input_st = ME.SparseTensor(
        #     features=torch.from_numpy(input_F),
        #     coordinates=torch.from_numpy(input_C),
        #     tensor_stride=in_tensor_stride,
        # )
        # input_st_sorted = input_st.F[sort_coo_coordinate(input_st.C)]
        mask = generate_mask(input_C, in_tensor_stride, input_h, input_w)

        # compute and sort output sparse tensor
        output_st = test_module(input_st)
        output_st_sorted = output_st.F[sort_coo_coordinate(output_st.C)]

        if args.save_dir:
            fpath_F = os.path.join(args.save_dir, npy_of(layer, 3, "output_integer"))
            fpath_C = os.path.join(
                args.save_dir, npy_of(layer, 3, f"output_coordinate_stride{out_tensor_stride}")
            )
            np.save(fpath_F, output_st.F)
            np.save(fpath_C, output_st.C)
            ipath_F = os.path.join(args.save_dir, npy_of(layer, 1, "input_integer"))
            ipath_C = os.path.join(
                args.save_dir, npy_of(layer, 1, f"input_coordinate_stride{in_tensor_stride}")
            )
            np.save(ipath_F, input_st.F)
            np.save(ipath_C, input_st.C)
            if residual:
                fpath = os.path.join(args.save_dir, npy_of(layer, 3, "Sid_bar"))
                np.save(fpath, module_kwargs["id_scale"])
        # tmp/block_2_conv3_Sid_bar.npy
        # output_st_sorted = output_st.F[sort_coo_coordinate(output_st.C)]
        # # load and sort ground truth
        # fpath = os.path.join(data_dir, npy_of(layer, 3, "output_integer"))
        # output_gt_F = np.load(fpath)
        # fpath = os.path.join(
        #     data_dir, npy_of(layer, 3, f"output_coordinate_stride{out_tensor_stride}")
        # )
        # output_gt_C = np.load(fpath)
        # output_gt_st_sorted = output_gt_F[
        #     sort_coo_coordinate(torch.from_numpy(output_gt_C))
        # ]
        # output_gt_st_sorted = np.round(output_gt_st_sorted)
        # # compare
        # diff = torch.Tensor.numpy(output_st_sorted) - output_gt_st_sorted
        # nz_idx_list = np.argwhere(diff != 0)
        # print("min diff:", diff.min(), "max diff:", diff.max())
        # print(f"number of wrong data:{nz_idx_list.shape[0]}/{(diff.size)}")
        # if nz_idx_list.shape[0] > 0:
        #     print(f"First 10 diff:")
        #     for i, nz_idx in enumerate(nz_idx_list[:10]):
        #         nz_idx = tuple(nz_idx)
        #         print(
        #             f"[{i}]: idx:{nz_idx}, gt:{output_gt_st_sorted[nz_idx]}, output:{output_st_sorted[nz_idx]}"
        #         )
        #     raise ValueError(f"output mismatch detected in layer {name}")
        # dump results and weights
        test_module.dump_results()
        test_module.dump_weights()
        # dump io data

        input_st_sorted = input_st.F[sort_coo_coordinate(input_st.C)]
        input_st_sorted = torch.round(input_st_sorted).type(torch.int)
        np.savetxt(
            os.path.join(out_dir, f"{legal_low(name)}_input.txt"),
            np.ndarray.flatten(input_st_sorted.numpy()),
            fmt="%d",
        )
        np.savetxt(
            os.path.join(out_dir, f"{legal_low(name)}_mask.txt"),
            np.ndarray.flatten(mask),
            fmt="%d",
        )
        output_st_sorted = torch.round(output_st_sorted).type(torch.int)
        np.savetxt(
            os.path.join(out_dir, f"{legal_low(name)}_output.txt"),
            np.ndarray.flatten(output_st_sorted.detach().numpy()),
            fmt="%d",
        )
    print("All layers passed tb test.")
    # dump tb data
    print("Dumping tb data...")
    shutil.copy(
        os.path.join(out_dir, f"{legal_low(cfg['layers'][0]['name'])}_input.txt"),
        os.path.join(out_dir, "tb_input_feature.txt"),
    )
    shutil.copy(
        os.path.join(out_dir, f"{legal_low(cfg['layers'][0]['name'])}_mask.txt"),
        os.path.join(out_dir, "tb_spatial_mask.txt"),
    )


if __name__ == "__main__":
    main()
