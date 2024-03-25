
import argparse, json, os
import torch
from torch.utils.data import DataLoader
from dataset import ThreeETplus_Eyetracking, ScaleLabel, TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
import numpy as np
import torch.nn.functional as F
import math
import MinkowskiEngine as ME
import tqdm


class KernelSparsityRecorder:
    def __init__(self, layer_number=5, file="kernel_sparsity.txt"):
        self.layer_num = layer_number
        self.count_num = 9
        self.amount = [[0 for _ in range(self.count_num)] for _ in range(self.layer_num)]
        self.total_amount = [0 for _ in range(self.layer_num)]
        self.file = file

    def update(self, coord, ts, idx):
        bs, h, w = coord[:, 0].unique().shape[0], int((coord[:, 2].max()/ts)+1), int((coord[:, 1].max()/ts)+1)
        mask = torch.zeros(bs, h, w)
        for c in coord:
            mask[int(c[0]), int(c[2]/ts), int(c[1]/ts)] = 1

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


class LayerSparsityAnalyzer:
    def __init__(self, file_path, sample_size, h, w, length):
        self.coord = {}
        self.sparse_ratio = []
        self.file_path = file_path
        self.length = length
        self.height, self.width = h, w
        self.sample_size = sample_size
        if os.path.exists(file_path):
            self.file = open(file_path, "a")
        else:
            self.file = open(file_path, "w")
            self.file.write("input\tconv1\tconv2\tconv3\tconv4\tconv5\n")

    def update(self, coord, idx):
        if idx in self.coord:
            self.coord[idx] = np.concatenate([self.coord[idx], coord], 0)
        else:
            self.coord[idx] = coord

    def process(self):
        # max_h, max_w = self.coord[0][:, 2].max()+1, self.coord[0][:, 1].max()+1
        for idx, (_, coord) in enumerate(self.coord.items()):
            total = math.ceil(self.height/math.pow(2, idx)) * math.ceil(self.width/math.pow(2, idx)) * \
                    self.sample_size * self.length
            self.sparse_ratio.append(str(round(coord.shape[0] / total, 4)))
        # with open(self.file_path, "w") as f:
        # self.file.write("{}\t{}\t".format(self.window, self.denoise))
        self.file.write("\t".join(self.sparse_ratio)+"\n")



# class SparsityAnalyzer:
#     def __init__(self, file_path, phase, height, width):
#         self.height, self.width = height, width
#         self.phase = phase
#         self.cnt_coord = 0
#         self.cnt_sample = 0
#         if os.path.exists(file_path):
#             self.file = open(file_path, "a")
#         else:
#             self.file = open(file_path, "w")
#             self.file.write("phase\tsparsity\n")
#
#     def update(self, coord, dense):
#         self.cnt_sample += dense.shape[0] * self.height * self.width
#         self.cnt_coord += coord.shape[0]
#
#     def release(self):
#         sparsity = 1 - self.cnt_coord / self.cnt_sample
#         self.file.write(f"{self.phase}\t{sparsity}\n")
#         self.file.close()


def main(args):
    phases = args.phase
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    height, width = int(args.sensor_height*args.spatial_factor), int(args.sensor_width * args.spatial_factor)

    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    conv = ME.MinkowskiChannelwiseConvolution(
            3,
            kernel_size=3,
            stride=2,
            dimension=2
        ).cuda()

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640 * factor, pseudo_height=480 * factor)
    ])

    slicing_time_window = args.train_length * int(10000 / temp_subsample_factor)  # microseconds
    train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)  # microseconds

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640 * factor), int(480 * factor), 2), \
                               n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    def dense_to_sparse(dense):
        non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense[select_indices], dim=-2)
        return non_zero_indices, features

    # sparsity_file = f"./metadata/3et_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}"  + "_sparsity.txt"
    sparsity_file = "layer_sparsity.txt"
    # file = open(sparsity_file, "w")
    layer_number = 4

    for phase in phases:
        assert phase in ["train", "val", "test"]
        data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split=phase,
                                              transform=transforms.Downsample(spatial_factor=factor),
                                              target_transform=label_transform)

        slicer = SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window - train_stride_time, \
                                                seq_length=args.train_length, seq_stride=args.train_stride,
                                                include_incomplete=False)
        data = SlicedDataset(data_orig, slicer, transform=post_slicer_transform,
                                   metadata_path=f"./metadata/3et_{phase}_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
        
        cached_path = f'./cached_dataset/{phase}_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}'
        data = DiskCachedDataset(data, cache_path=cached_path)
        loader = DataLoader(data, batch_size=1, shuffle=False,  num_workers=0)

        Sparsity = LayerSparsityAnalyzer(sparsity_file, len(loader), height, width, args.train_length)
        layer_sparsity = KernelSparsityRecorder(layer_number=layer_number)

        loader_descend = tqdm.tqdm(loader, desc=f"{phase} phase")
        for inputs, targets in loader_descend:
            # continue
            tensor_stride = 1
            spatial_idx = 0
            (b, l, c, w, h) = inputs.shape
            inputs = inputs.view(b*l, c, w, h)
            inputs = inputs.permute(0, 2, 3, 1)
            coord, features = dense_to_sparse(inputs)
            x = ME.SparseTensor(
                coordinates=coord.int(), features=features, device="cuda:0"
            )
            Sparsity.update(x.C.cpu().numpy(), 0)
            layer_sparsity.update(x.C.cpu(), tensor_stride, 0)
            spatial_idx += 1

            for i in range(layer_number):
                tensor_stride *= 2
                x = conv(x)
                Sparsity.update(x.C.cpu().numpy(), i+1)
                if i != layer_number - 1:
                    layer_sparsity.update(x.C.cpu(), tensor_stride, i+1)
                spatial_idx += 1
            # batch_idx += 1
        Sparsity.process()
        layer_sparsity.release()
        recorder.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument('--phase', default=["test"], nargs='+', type=str, help='Frame delay list')

    args = parser.parse_args()

    main(args)
