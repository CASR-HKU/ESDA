import numpy as np
from tonic import DiskCachedDataset
from .augmentation import Flip, Shift, DropEvent, TemporalMask, SpatialMask
import json
from .sample import generate_heatmap_label
import tonic.transforms as transforms
from .custom_transforms import SliceLongEventsToShort, EventSlicesToVoxelGrid
from tonic import SlicedDataset
import torch

aug_funcs = {
    'flip': Flip,
    'shift': Shift,
    'temporal_mask': TemporalMask,
    "spatial_mask": SpatialMask,
    # "drop_event": DropEvent
}


class EyeCenterRegressionSliceDataset:
    def __init__(self, args, data_orig, slicer, metadata_path, istrain=False, use_upsample=False, return_meta=True):
        self.istrain = istrain
        self.use_upsample = use_upsample
        self.return_meta = return_meta

        augment_path = None
        if "augment_path" in args:
            augment_path = args.augment_path

        factor = args.spatial_factor # spatial downsample factor
        temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

        transformer = [
            SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0,
                                   include_incomplete=True),
            EventSlicesToVoxelGrid(sensor_size=(int(640 * factor), int(480 * factor), 2), \
                                   n_time_bins=args.n_time_bins,
                                   per_channel_normalize=args.voxel_grid_ch_normaization)
        ]

        if augment_path is None:
            self.transforms = []
        else:
            aug_cfg = self.parse_cfg(augment_path)
            if "drop_event" in aug_cfg:
                if istrain:
                    transformer = [transforms.DropEvent(p=aug_cfg["drop_event"]["ratio"])] + transformer
                aug_cfg.pop("drop_event")
            self.transforms = [
                aug_funcs[func](**kwargs) for func, kwargs in aug_cfg.items()
            ]


        post_slicer_transform = transforms.Compose(transformer)

        # cached_path = f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}'
        self.data = SlicedDataset(data_orig, slicer, transform=post_slicer_transform, metadata_path=metadata_path)

    def parse_cfg(self, cfg_file):
        with open(cfg_file, 'r') as cfg:
            cfg = json.load(cfg)
        return cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        frames, target = self.data[item]
        if self.istrain:
            for transform in self.transforms:
                frames, target = transform(frames, target)

        if self.use_upsample:
            t, c, h, w = frames.shape
            real_kps = np.zeros_like(target)
            real_kps[:, 0] = (target[:, 0] * w).astype(np.int_)
            real_kps[:, 1] = (target[:, 1] * h).astype(np.int_)
            hms = generate_heatmap_label(real_kps, h, w)
            return frames, (target, hms.astype(np.float32))
        else:
            if self.return_meta:
                dataset_index, slice_index = self.data.slice_dataset_map[item]
                data, targets = self.data.dataset[dataset_index]
                data_slice, target_slice = self.data.slicer.slice_with_metadata(
                    data, targets, [self.data.metadata[dataset_index][slice_index]]
                )
                return frames, target, {"event": data_slice[0], "index": (dataset_index, slice_index)}
            else:
                return frames, target


class EyeCenterRegressionDataset:
    def __init__(self, data, cache_path, augment_path=None, istrain=False, use_upsample=False):
        self.data = DiskCachedDataset(data, cache_path=cache_path)
        self.use_upsample = use_upsample
        self.istrain = istrain
        self.mask_temporal = False
        if augment_path is None:
            self.transforms = []
        else:
            self.transforms = [
                aug_funcs[func](**kwargs) for func, kwargs in self.parse_cfg(augment_path).items()
            ]

    def __len__(self):
        return len(self.data)

    def parse_cfg(self, cfg_file):
        with open(cfg_file, 'r') as cfg:
            cfg = json.load(cfg)
        self.mask_temporal = False if "temporal_mask" not in cfg else True
        return cfg

    def __getitem__(self, item):
        frames, target = self.data[item]
        if self.istrain:
            for transform in self.transforms:
                frames, target = transform(frames, target)

        if self.use_upsample:
            t, c, h, w = frames.shape
            real_kps = np.zeros_like(target)
            real_kps[:, 0] = (target[:, 0] * w).astype(np.int_)
            real_kps[:, 1] = (target[:, 1] * h).astype(np.int_)
            hms = generate_heatmap_label(real_kps, w, h)
            return frames, (target, hms.astype(np.float32))
        else:
            masked_temporal_length = torch.tensor(0)
            if self.mask_temporal and self.istrain:
                for transform in self.transforms:
                    if isinstance(transform, TemporalMask):
                        masked_temporal_length = transform.masked_temporal_length
            return frames, target, masked_temporal_length


