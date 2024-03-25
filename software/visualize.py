
import argparse, json, os

import cv2
import torch
from dataset.histogram import histogram_visualize
from dataset import ThreeETplus_Eyetracking, ScaleLabel, TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets, to_frame_numpy
from dataset.histogram import HistogramGenerator
import tonic.transforms as transforms
from tonic import SlicedDataset
import numpy as np
from model import *
import math
from utils.utils import merge_images
from dataset.regression_dataset import EyeCenterRegressionDataset, EyeCenterRegressionSliceDataset


from utils.utils import merge, draw_points, draw_distance, process, merge_images


def main(args):
    checkpoint = args.checkpoint
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

    if checkpoint:
        model = eval(args.architecture)(args).to(args.device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640 * factor, pseudo_height=480 * factor)
    ])

    slicing_time_window = args.train_length * int(10000 / temp_subsample_factor)  # microseconds
    train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)  # microseconds


    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
    # post_slicer_transform = transforms.Compose([
        # SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
        # transforms.toTensor(),
        # transforms.ToFrame(sensor_size=(int(640 * factor), int(480 * factor), 2), n_time_bins=args.train_length, overlap=0, include_incomplete=True),
        # EventSlicesToVoxelGrid(sensor_size=(int(640 * factor), int(480 * factor), 2), \
        #                        n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    # ])

    for phase in phases:
        assert phase in ["train", "val", "test"]
        data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split=phase,
                                              transform=transforms.Downsample(spatial_factor=factor),
                                              target_transform=label_transform)

        slicer = SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window - train_stride_time, \
                                                seq_length=args.train_length, seq_stride=args.train_stride,
                                                include_incomplete=False)
        length, stride = (args.train_length, args.train_stride) \
            if phase == "train" else (args.val_length, args.val_stride)
        data = EyeCenterRegressionSliceDataset(args, data_orig, slicer, return_meta=True,
                                               metadata_path=f"./metadata/3et_{phase}_vl_{length}_vs{stride}_ch{args.n_time_bins}", istrain=False, use_upsample=False)

        # data = SlicedDataset(data_orig, slicer,
        #                      metadata_path=f"./metadata/3et_{phase}_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
        # loader = DataLoader(data, batch_size=1, shuffle=False,  num_workers=0)
        # post_slicer_transform = transforms.Compose([
        #     SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        #     EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
        #                             n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
        # ])
        to_hist = HistogramGenerator(height=height, width=width)
        with torch.no_grad():
            for i in range(len(data)):
                frame, targets, raw_events = data[i]
                events = np.concatenate((np.expand_dims(raw_events["event"]["x"], axis=1),
                                         np.expand_dims(raw_events["event"]["y"], axis=1),
                                         np.expand_dims(raw_events["event"]["t"], axis=1),
                                         np.expand_dims(raw_events["event"]["p"], axis=1)), axis=1)

                imgs, raw_imgs = [], []
                start_time = raw_events["event"][0][0]
                for t in range(args.train_length):
                    time_upper, time_lower = t * 50000 + start_time, (t + 1) * 50000 + start_time
                    clipped_events = events[(events[:, 2] >= time_upper) & (events[:, 2] < time_lower)]
                    # print(clipped_events.shape[0])
                    event_num = clipped_events.shape[0]

                    histogram = to_hist.to_histogram(clipped_events.astype(np.float32))
                    histogram = torch.from_numpy(histogram)
                    histogram = torch.nn.functional.interpolate(histogram.permute(2, 0, 1).unsqueeze(0),
                                                                size=torch.Size((height, width)))
                    histogram = histogram.squeeze(0).permute(1, 2, 0)
                    histogram = torch.tensor(histogram, dtype=torch.float32).unsqueeze(0)
                    img, raw_img = histogram_visualize(histogram, (height, width), [str(event_num)])
                    imgs.append(img)
                    raw_imgs.append(raw_img)

                if args.save_dir:
                    for img_idx, img in enumerate(raw_imgs):
                        cv2.imwrite(os.path.join(args.save_dir, f"{i}_{img_idx}.png"), img[0])

                targets = process(targets[:, :2], height, width)
                inputs = draw_points(imgs, targets, (0, 0, 255))

                if checkpoint:
                    outputs = model(torch.tensor(frame).unsqueeze(dim=0).to(args.device))
                    outputs = process(outputs.squeeze().numpy(), height, width)
                    inputs = draw_points(inputs, outputs, (0, 0, 0))
                    distances = np.linalg.norm(targets - outputs, axis=1)
                    # inputs = draw_distance(inputs, distances, (0, 0, 0))

                merged_image = merge_images(inputs, (height, width))
                # output = merge(inputs, height, width)

                cv2.imshow("histogram", cv2.resize(merged_image, (1080, 720)))
                cv2.waitKey(0)
            # a = 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Use device")
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument('--phase', default=["train"], nargs='+', type=str, help='Frame delay list')
    parser.add_argument("--checkpoint", type=str, default="", help="path to JSON configuration file")
    parser.add_argument("--save_dir", type=str, default="save", help="path to JSON configuration file")

    args = parser.parse_args()
    main(args)
