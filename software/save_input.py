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

    height, width = int(args.sensor_height * args.spatial_factor), int(args.sensor_width * args.spatial_factor)

    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

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

    os.makedirs(os.path.join(args.output_dir, "feat"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "out"), exist_ok=True)

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
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

        loader_descend = tqdm.tqdm(loader, desc=f"{phase} phase")
        idx = 0

        for inputs, targets in loader_descend:
            # continue
            (b, l, c, w, h) = inputs.shape
            inputs = inputs.view(b * l, c, w, h)
            np.save(os.path.join(args.output_dir, "feat/{}.npy".format(idx)), inputs.numpy())

            inputs = inputs.permute(0, 2, 3, 1)
            coord, features = dense_to_sparse(inputs)
            x = ME.SparseTensor(
                coordinates=coord.int(), features=features, device="cuda:0"
            )

            mask = torch.zeros(l, height, width)
            for c in x.C:
                mask[c[0], c[1], c[2]] = 1

            # inputs = inputs.view(b, l, c, w, h)
            mask = mask.view(l, 1, height, width)
            np.save(os.path.join(args.output_dir, "mask/{}.npy".format(idx)), mask.numpy())
            np.save(os.path.join(args.output_dir,  "out/{}.npy".format(idx)), np.array(targets))
            idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument('--phase', default=["test"], nargs='+', type=str, help='Frame delay list')
    parser.add_argument('--output_dir', type=str, default="output", help="path to data directory")

    args = parser.parse_args()

    main(args)
