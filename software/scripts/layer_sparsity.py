import math

import os
import sys
import numpy as np
import torch

import MinkowskiEngine as ME
import argparse
import yaml
from dataset.loader import Loader
import utils.utils as utils
from collections import defaultdict
import torch.nn.functional as F
from models.drop_utils import RandomDrop


class LayerSparsityRecorder:
    def __init__(self, layer_number=5, file="DVS_ratio.txt"):
        self.layer_num = layer_number
        self.count_num = 9
        self.amount = [[0 for _ in range(self.count_num)] for _ in range(self.layer_num)]
        self.total_amount = [0 for _ in range(self.layer_num)]
        self.file = file

    def update(self, coord, ts, idx):
        bs, h, w = coord[:, 0].unique().shape[0], int((coord[:, 2].max()/ts[0])+1), int((coord[:, 1].max()/ts[0])+1)
        mask = torch.zeros(bs, h, w)
        for c in coord:
            mask[int(c[0]), int(c[2]/ts[0]), int(c[1]/ts[1])] = 1

        pooled_mask = F.max_pool2d(mask.unsqueeze(1), 2, 2)
        conved_masked = F.conv2d(pooled_mask, torch.ones(1, 1, 3, 3), stride=1)
        self.total_amount[idx] += ((conved_masked != 0).sum()).tolist()
        for i in range(self.count_num):
            self.amount[idx][i] += ((conved_masked == (i+1)).sum()).tolist()

        return

    def release(self):
        ratio = [[0 for _ in range(self.count_num)] for _ in range(self.layer_num)]
        for layer_idx in range(self.layer_num):
            for idx in range(self.count_num):
                ratio[layer_idx][idx] = round(self.amount[layer_idx][idx] / self.total_amount[layer_idx], 4)
                # self.amount[layer_idx][idx] /= self.total_amount
        print(ratio)
        if self.file:
            with open(self.file, "a") as f:
                f.write(" ".join([str(r) for r in ratio]) + "\n")
        return



class SparsityAnalyzer:
    def __init__(self, file_path, cfg, sample_size):
        self.coord = {}
        self.sparse_ratio = []
        self.file_path = file_path
        self.sample_size = sample_size
        self.window, self.denoise = cfg
        if os.path.exists(file_path):
            self.file = open(file_path, "a")
        else:
            self.file = open(file_path, "w")
            self.file.write("window\tdenoise\tinput\tconv1\tconv2\tconv3\tconv4\tconv5\n")

    def update(self, coord, idx):
        if idx in self.coord:
            self.coord[idx] = np.concatenate([self.coord[idx], coord], 0)
        else:
            self.coord[idx] = coord

    def process(self):
        max_h, max_w = self.coord[0][:, 2].max()+1, self.coord[0][:, 1].max()+1
        for idx, (_, coord) in enumerate(self.coord.items()):
            total = max_h * max_w * self.sample_size / math.pow(2, idx) / math.pow(2, idx)
            self.sparse_ratio.append(str(round(coord.shape[0] / total, 4)))
        # with open(self.file_path, "w") as f:
        self.file.write("{}\t{}\t".format(self.window, self.denoise))
        self.file.write("\t".join(self.sparse_ratio)+"\n")


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--output_file', help='Path to output txt')
    parser.add_argument('--data_path', default="", help='Path to dataset')
    parser.add_argument('--raw_size', action='store_true', help='Use raw size as the input size')
    parser.add_argument('--min_event', type=int, default=0)
    parser.add_argument('--drop_ratio', default=0, type=float)

    args = parser.parse_args()

    settings_filepath = args.settings_file
    target_num = -1
    nr_input_channels = 2

    conv = ME.MinkowskiChannelwiseConvolution(
            nr_input_channels,
            kernel_size=3,
            stride=2,
            dimension=2
        )

    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    if args.data_path != "":
        cfg["dataset"]["dataset_path"] = args.data_path
    Dataset = utils.select_dataset(cfg["dataset"]["name"])
    drop_unit = RandomDrop(drop_config=[args.drop_ratio, False])

    layer_number = 5
    if args.debug:
        windows = [2048]
        denoises = [-1]
    else:
        if "Gesture" in cfg["dataset"]["name"] or "DVSPreprocessed" in cfg["dataset"]["name"]:
            # DVS config
            windows = [2048]
            denoises = [-1]
        elif "ASL" in cfg["dataset"]["name"]:
            windows = [1024, 512, 256, 128]
            denoises = [-1]
        elif "NMNIST" in cfg["dataset"]["name"]:
            windows = [1024]
            denoises = [-1]
            layer_number = 3
        elif "Cal" in cfg["dataset"]["name"]:
        # NCal config
            windows = [50000, 30000, 10000, 5000]
            denoises = [-1]
        else:
            windows = [256]
            denoises = [-1]
            layer_number = 3

    file_path = args.output_file
    if os.path.exists(file_path):
        os.remove(file_path)

    idx = 0
    for window in windows:
        for denoise in denoises:
            print("Processing {}: window: {}, denoise: {}".format(idx, window, denoise))
            cfg["transform"]["sample"]["window"] = window
            try:
                cfg["transform"]["denoise"]["filter_time"] = denoise
            except:
                cfg["transform"] = {}
                cfg["transform"]["denoise"] = {}
                cfg["transform"]["denoise"]["filter_time"] = denoise
            val_dataset = Dataset(cfg, mode="validation", cross_valid=-1, shuffle=False, min_event=args.min_event)

            loader = Loader(val_dataset, 128, device="cuda:0", num_workers=0, pin_memory=False,
                            shuffle=False)
            conv = conv.cuda().eval()
            Sparsity = SparsityAnalyzer(file_path, (window, denoise), len(val_dataset))
            layer_sparsity = LayerSparsityRecorder(layer_number=layer_number)
            idx += 1
            if idx == target_num:
                print("Finish with {} samples".format(target_num))
                sys.exit(0)

            batch_idx = 0
            for i_batch, sample_batched in enumerate(loader):
                spatial_idx = 0
                x = ME.SparseTensor(
                    coordinates=sample_batched["coordinates"], features=sample_batched["features"], device="cuda:0"
                )
                Sparsity.update(x.C.cpu().numpy(), 0)
                layer_sparsity.update(x.C.cpu(), x.tensor_stride, 0)
                spatial_idx += 1

                for i in range(layer_number):
                    x = conv(x)
                    if i > 0 and args.drop_ratio > 0:
                        x = drop_unit(x)[0]
                    Sparsity.update(x.C.cpu().numpy(), i+1)
                    if i != layer_number - 1:
                        layer_sparsity.update(x.C.cpu(), x.tensor_stride, i+1)
                    spatial_idx += 1
                batch_idx += 1
            Sparsity.process()
            layer_sparsity.release()


if __name__ == '__main__':
    main()
