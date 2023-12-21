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
    assert C % PF == 0
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


def pack_1x1_weights(weights, PI, precision=8):
    assert precision == 8
    weights = weights.astype(int)
    OC, IC = weights.shape
    assert IC % PI == 0
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
        print(i, " scale:", scale[i], " bias:", bias[i], output_array[i])
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

    def get_flatten_weights(self):
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
        print("scale:", scale)
        print("bias:", bias)
        scale_bias = pack_scale_bias(scale, bias, self.bias_bits)
        save_array_as_c_init(
            scale_bias, os.path.join(self.out_dir, self.prefix + "_s.txt")
        )
        save_array_as_c_init(
            packed_weights,
            os.path.join(self.out_dir, self.prefix + "_w.txt"),
        )

    def get_flatten_results(self):
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
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -127, 127)
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

    def get_flatten_weights(self):
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

    def get_flatten_results(self):
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

        print("mul:", mul[sort_coo_coordinate(self.psum_coords)])
        mul = (mul) >> self.shift_n

        int_identity = torch.round(self.id_scale) * conv1_int_input
        int_identity = int_identity.type(torch.int64)
        int_identity += round_shift

        int_identity = (int_identity) >> self.shift_n

        id_flatten = (
            int_identity[sort_coo_coordinate(self.psum_coords)].numpy().flatten()
        )
        print("int_identity:", int_identity[sort_coo_coordinate(self.psum_coords)])
        mul_flatten = mul[sort_coo_coordinate(self.psum_coords)].numpy().flatten()

        self.psum_mult_shift = mul + int_identity

        self.psum_mult_shift = self.psum_mult_shift.type(torch.FloatTensor)
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -127, 127)
        self.layer_output = self.psum_mult_shift_clamp
        output_sparse_tensor = ME.SparseTensor(
            features=self.psum_mult_shift_clamp,
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

    def get_flatten_weights(self):
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

    def get_flatten_results(self):
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
        self.psum_mult_shift_clamp = torch.clamp(self.psum_mult_shift, -127, 127)
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
            )

    def dump_weights(self):
        self.module_1x1.get_flatten_weights()
        self.module_3x3_dw.get_flatten_weights()
        self.module_1x1_2.get_flatten_weights()
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
        self.module_1x1.get_flatten_results()
        self.module_3x3_dw.get_flatten_results()
        self.module_1x1_2.get_flatten_results()

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
    coordinate = coordinate // tensor_stride
    mask = np.zeros(
        (input_height // tensor_stride, input_width // tensor_stride), dtype=np.uint8
    )
    mask[coordinate[:, 1], coordinate[:, 2]] = 1
    return mask


def gen_mask(height, width, nz_ratio, save_path):
    mask = np.zeros((height * width), dtype=np.uint8)
    nz_num = int(height * width * nz_ratio)
    nz_idx = np.random.choice(height * width, nz_num, replace=False)
    mask[nz_idx] = 1
    np.savetxt(save_path, mask.flatten(), fmt="%d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", "-d", type=str, default=".")
    args = parser.parse_args()

    # load cfg
    if not os.path.exists(os.path.join(args.work_dir, "cfg.json")):
        raise FileNotFoundError("cfg.json not found.")
    cfg = json.load(open(os.path.join(args.work_dir, "cfg.json")))
    # check out_dir
    out_dir = os.path.join(args.work_dir, "data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # check data_dir
    data_dir = cfg["model_path"]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir: {data_dir} not found.")

    # parse cfg
    shift_n = 32
    input_h, input_w = cfg["input_shape"]
    # for each layer
    for layer_i, layer in enumerate(cfg["layers"]):
        name = legal_low(layer["name"])
        parallelism = layer["parallelism"]
        if layer["type"] == "conv":
            residual = False
            stride = layer["stride"]
            in_tensor_stride = layer["tensor_stride"]
            out_tensor_stride = layer["tensor_stride"]
            raise NotImplementedError("conv layer not supported yet.")
        elif layer["type"] == "block":
            residual = layer["residual"]
            stride = layer["stride"]
            in_tensor_stride = layer["tensor_stride"][0]
            out_tensor_stride = layer["tensor_stride"][-1]
        else:
            raise ValueError(f"Unknown layer type: {layer['type']}")
        # build module kwargs
        module_kwargs = {}
        for i in range(3):
            for dtype, fname in zip(
                ["weight", "bias", "scale"],
                ["weight_integer", "bias_integer", "Sw_bar"],
            ):
                k = f"{dtype}_{i}"
                fpath = os.path.join(data_dir, npy_of(name, i + 1, fname))
                module_kwargs[k] = np.load(fpath)
        if residual:
            fpath = os.path.join(data_dir, npy_of(name, 3, "Sid_bar"))
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
        # build module
        test_module = tb_1x1_3x3dw_1x1_block(**module_kwargs)
        # load input
        fpath = os.path.join(data_dir, npy_of(name, 1, "input_integer"))
        input_F = np.load(fpath)
        fpath = os.path.join(
            data_dir, npy_of(name, 1, f"input_coordinate_stride{in_tensor_stride}")
        )
        input_C = np.load(fpath)
        # generate mask
        mask = generate_mask(input_C, in_tensor_stride, input_h, input_w)
        # build input sparse tensor
        input_st = ME.SparseTensor(
            features=torch.from_numpy(input_F),
            coordinates=torch.from_numpy(input_C),
            tensor_stride=in_tensor_stride,
        )
        input_st_sorted = input_st.F[sort_coo_coordinate(input_st.C)]

        # compute and sort output sparse tensor
        output_st = test_module(input_st)
        output_st_sorted = output_st.F[sort_coo_coordinate(output_st.C)]
        # load and sort ground truth
        fpath = os.path.join(data_dir, npy_of(name, 3, "output_integer"))
        output_gt_F = np.load(fpath)
        fpath = os.path.join(
            data_dir, npy_of(name, 3, f"output_coordinate_stride{out_tensor_stride}")
        )
        output_gt_C = np.load(fpath)
        output_gt_st_sorted = output_gt_F[
            sort_coo_coordinate(torch.from_numpy(output_gt_C))
        ]
        output_gt_st_sorted = np.round(output_gt_st_sorted)
        # compare
        diff = output_st_sorted - torch.from_numpy(output_gt_st_sorted)
        print("min diff:", diff.min(), "max diff:", diff.max())
        print(f"number of wrong data:{np.count_nonzero(diff)}/{torch.numel(diff)}")
        if np.count_nonzero(diff) > 0:
            raise ValueError(f"output mismatch detected in layer {name}")
        # dump results and weights
        test_module.dump_results()
        test_module.dump_weights()
        # dump io data
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
    # dump tb data
    shutil.copy(
        os.path.join(out_dir, f"{legal_low(cfg['layers'][0]['name'])}_input.txt"),
        os.path.join(out_dir, "tb_input_feature.txt"),
    )
    shutil.copy(
        os.path.join(out_dir, f"{legal_low(cfg['layers'][0]['name'])}_mask.txt"),
        os.path.join(out_dir, "tb_spatial_mask.txt"),
    )
    shutil.copy(
        os.path.join(out_dir, f"{legal_low(cfg['layers'][-1]['name'])}_output.txt"),
        os.path.join(out_dir, "tb_output.txt"),
    )
    # gen mask
    for nz_ratio in np.arange(0.1, 1.1, 0.1):
        nz_ratio = round(nz_ratio, 1)
        gen_mask(
            *cfg["layers"][0]["input_shape"],
            nz_ratio,
            os.path.join(out_dir, f"mask_{nz_ratio}.txt"),
        )


if __name__ == "__main__":
    main()
