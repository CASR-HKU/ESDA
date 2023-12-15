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
    "NCAL": 11831,
    "NMNIST": 19942,
    "Roshambo": 205695,
}


class inverted_residual_block:
    def __init__(self, bitstream, Height, Width, IC, OC, mask_bits):
        self.bitstream = bitstream
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
            shape=(self.Height * self.Width * self.OC), dtype=np.int8
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

    def load_mask(self, mask_path):
        assert os.path.exists(mask_path), f"mask_path={mask_path} does not exist"
        mask = np.load(mask_path).astype(int)
        self.mask_data = self.pad_mask(mask)
        packed_mask = self.pack_mask(self.mask_data)
        for i in range(packed_mask.shape[0]):
            self.mask_buffer[i] = packed_mask[i]
        self.mask_buffer.flush()

    def run(self, dataset, num_run, verbose=False):
        total_time = 0
        dataset_path = os.path.join(
            "/home/xilinx/jupyter_notebooks/event_dataset", dataset
        )
        range_bound = dataset_size[dataset] if num_run <= 0 else num_run
        print(f"dataset={dataset}, num_run={num_run}, range_bound={range_bound}")
        print(f"mask_bits={self.mask_bits}")
        mask_idx_list = []
        if dataset == "Roshambo":
            while len(mask_idx_list) < range_bound:
                random_mask = np.random.randint(0, dataset_size[dataset])
                mask_path = os.path.join(dataset_path, f"{random_mask}.npy")
                if os.path.exists(mask_path):
                    mask_idx_list.append(random_mask)
        else:
            mask_idx_list = np.random.randint(0, dataset_size[dataset], range_bound)
        print(f"len(mask_idx_list)={len(mask_idx_list)}")
        # loop over all test data
        run_cnt = 0
        start_time = time.time()
        for mask_idx in tqdm(mask_idx_list, mininterval=30, maxinterval=60, miniters=1):
            mask_path = os.path.join(dataset_path, f"{mask_idx}.npy")
            if not os.path.exists(mask_path):
                continue
            self.load_mask(mask_path)
            num_nz = np.count_nonzero(self.mask_data)
            self.accel.register_map.num_nz = int(num_nz)
            idle = 0
            begin = time.time()
            self.accel.register_map.CTRL.AP_START = 1
            while idle == 0:
                idle = self.accel.register_map.CTRL.AP_IDLE
            end = time.time()
            tmp_time = (end - begin) * 1000
            total_time += tmp_time
            run_cnt += 1
        end_time = time.time()
        print(
            f"hw loop time(s): {end_time:.4f} - {start_time:.4f}"
            + f" = {end_time - start_time:.4f} / {run_cnt}"
            + f" = {(end_time - start_time)/(run_cnt):.4f}"
        )
        print(f"hw average runtime(ms): {total_time / run_cnt}")

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
    hw = inverted_residual_block(hw_path, top_ih, top_iw, top_ic, top_oc, mask_bits)
    hw.run(dataset, args.num_run, verbose=True)
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
