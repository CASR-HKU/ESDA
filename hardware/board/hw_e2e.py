import warnings
import pynq  # type: ignore
from pynq import Overlay  # type: ignore
from pynq import allocate  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy as np
import time
import math
import json
import os
import argparse
import subprocess
import signal


valid_dataset_list = ["ASL", "DVS", "NCAL", "NMNIST", "Roshambo"]

dataset_size = {
    "ASL": 20160,
    "DVS": 6959,
    "NCAL": 2612,
    "NMNIST": 19942,
    "Roshambo": 205695,
}

classes_size = {
    "ASL": 25,
    "DVS": 10,
    "NCAL": 101,
    "NMNIST": 10,
    "Roshambo": 4,
}

input_shape = {
    "ASL": (180, 240),
    "DVS": (128, 128),
    "NCAL": (180, 240),
    "NMNIST": (34, 34),
    "Roshambo": (64, 64),
}


class inverted_residual_block:
    def __init__(self, bitstream, Height, Width, IC, OC, mask_bits, dataset):
        self.bitstream = bitstream
        self.dataset = dataset
        self.input_shape = input_shape[dataset]
        self.overlay = Overlay(self.bitstream)
        self.accel = self.overlay.top_0
        self.overlay.download()
        self.Height = Height
        self.Width = Width
        self.IC = IC
        self.OC = OC
        self.mask_bits = mask_bits
        self.input_feature_buffer = allocate(
            shape=(self.Height * self.Width * self.IC), dtype=np.int8
        )
        self.output_feature_buffer = allocate(
            shape=16, dtype=np.int32
            # shape=(self.Height * self.Width * self.OC), dtype=np.int32
        )
        self.mask_buffer = allocate(
            shape=(
                self.Height
                * math.ceil(self.Width / self.mask_bits)
                * self.mask_bits
                // 8
            ),
            dtype=np.uint8,
        )

        self.accel.register_map.act_in_1 = self.input_feature_buffer.physical_address
        self.accel.register_map.act_out_1 = self.output_feature_buffer.physical_address
        self.accel.register_map.mask_1 = self.mask_buffer.physical_address

    def load_feat_npy(self, feat_path):
        print("Loading feature npy")
        print(feat_path)
        feat = np.load(feat_path)
        # print(feat)
        for i in range(feat.shape[0]):
            self.input_feature_buffer[i] = int(feat[i])
        self.input_feature_buffer.flush()

    def load_mask_npy(self, mask_path):
        print("Loading mask npy")
        print(mask_path)
        mask = np.load(mask_path).astype(int).reshape(self.input_shape[0], self.input_shape[1])        
        self.mask_data = self.pad_mask(mask)
        print(self.mask_data.shape)
        packed_mask = self.pack_mask(self.mask_data)
        print(packed_mask.shape)
        for i in range(packed_mask.shape[0]):
            self.mask_buffer[i] = packed_mask[i]
        self.mask_buffer.flush()

    def pack_mask(self, mask):
        mask_flattened = mask.flatten().astype(int)
        byte_size = math.ceil(mask_flattened.shape[0] / self.mask_bits) * (
            self.mask_bits // 8
        )
        mask_buffer = np.zeros(byte_size, dtype=np.int8)

        index = 0
        for i in range(math.ceil(mask_flattened.shape[0] / 8)):
            pack = 0
            for j in range(8):
                pack += mask_flattened[index] << j
                index += 1
                if index == mask_flattened.shape[0]:
                    break
            mask_buffer[i] = pack
        return mask_buffer

    def pad_mask(self, mask):
        padded_shape = (
            mask.shape[0],
            math.ceil(mask.shape[1] / self.mask_bits) * self.mask_bits,
        )
        mask_padded = np.pad(
            mask,
            (
                (0, 0),
                (0, padded_shape[1] - mask.shape[1]),
            ),
            "constant",
        )
        return mask_padded.flatten()

    def run(self, dataset, num_run, feat_path, mask_path, verbose=False):
        valid_classes = classes_size[dataset]
        total_time = 0

        self.load_feat_npy(feat_path)
        self.load_mask_npy(mask_path)
        # label = np.load(label_path)
        num_nz = np.count_nonzero(self.mask_data)
        self.accel.register_map.num_nz = int(num_nz)
        idle = 0
        self.accel.register_map.CTRL.AP_START = 1
        while idle == 0:
            idle = self.accel.register_map.CTRL.AP_IDLE
        out = self.output_feature_buffer
        out = out[:valid_classes]
        predict = np.argmax(out)

        print(f"predict={predict:3d}, out={out}")



    def check_output(self):
        flag = True
        for i in range(self.output_data.shape[0]):
            if self.output_data[i] != self.output_feature_buffer[i]:
                print(
                    "wrong at {} with gt:{}, out:{}".format(
                        i, self.output_data[i], self.output_feature_buffer[i]
                    )
                )
                flag = False
        if flag:
            print("pass tb!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_run", type=int)
    parser.add_argument("--work_dir", "-d", type=str, default=".")
    parser.add_argument("--enable_pm", action="store_true")
    args = parser.parse_args()

    # file path
    hw_path = os.path.join(args.work_dir, "top.bit")
    cfg_path = os.path.join(args.work_dir, "cfg.json")
    input_feat_path = os.path.join(args.work_dir, "tb_input_feature.npy")
    input_mask_path = os.path.join(args.work_dir, "tb_spatial_mask.npy")

    # load cfg
    cfg = json.load(open(cfg_path))
    top_ih = cfg["layers"][0]["input_shape"][0]
    top_iw = cfg["layers"][0]["input_shape"][1]
    top_ic = cfg["layers"][0]["channels"][0]
    top_oc = cfg["layers"][-1]["channels"][-1]
    mask_bits = 64 if cfg["dataset"] == "Roshambo" else 128
    dirname = os.path.basename(os.path.realpath(args.work_dir))
    print(dirname)
    try:
        dataset = dirname.split("-")[1].split("_")[0]
    except IndexError:
        dataset = cfg["dataset"]
        warnings.warn(
            f"cannot parse dataset from dirname={dirname}" + f", use {dataset}"
        )
    assert dataset in valid_dataset_list

    # start power monitor and wait for 1s to initialize
    if args.enable_pm:
        pm = subprocess.Popen(
            ["python3", "../power_monitor.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1)
    # start hw
    hw = inverted_residual_block(hw_path, top_ih, top_iw, top_ic, top_oc, mask_bits, dataset)
    hw.run(dataset, args.num_run, input_feat_path, input_mask_path, verbose=True)
    # send SIGINT to pm and wait for 10s to kill
    if args.enable_pm:
        pm.send_signal(signal.SIGINT)
        try:
            outs, errs = pm.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            pm.kill()
            outs, errs = pm.communicate()
            print("pm killed.")
        else:
            print("pm exited normally.")
        finally:
            print(f"pm stdout:")
            print(outs.decode("utf8"))
            print(f"pm stderr:")
            print(errs.decode("utf8"))


if __name__ == "__main__":
    main()
