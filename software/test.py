"""
Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, json, os, mlflow, csv
import torch
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from dataset.regression_dataset import EyeCenterRegressionSliceDataset


def main(args):
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

    # also dump the args to a JSON file in MLflow artifact
    # with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
    #     json.dump(vars(args), f)
    args.evaluate = True
    # Define your model, optimizer, and criterion
    model = eval(args.architecture)(args).to(args.device)
    model.eval()

    # test data loader always cuts the event stream with the labeling frequency
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    test_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="test", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.val_length*int(10000/temp_subsample_factor) #microseconds

    test_slicer = SliceByTimeEventsTargets(slicing_time_window,
                                           overlap=(slicing_time_window/args.val_length*(args.val_length-1)),
                                           seq_length=args.val_length, seq_stride=1, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])
    test_data = EyeCenterRegressionSliceDataset(args, test_data_orig, test_slicer, \
                    metadata_path=f"./metadata/3et_test_tl_{args.val_length}_ts1_ch{args.n_time_bins}", istrain=False, use_upsample=False)

    # test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform)

    # Uncomment the following lines to use the cached dataset
    # Use with caution! Don't forget to update the cache path if you change the dataset or the slicing parameters

    # test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform, \
    #     metadata_path=f"./metadata/3et_test_l{args.test_length}s{args.test_stride}_ch{args.n_time_bins}")

    # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
    # test_data = DiskCachedDataset(test_data, \
    #                               cache_path=f'./cached_dataset/test_l{args.test_length}s{args.test_stride}_ch{args.n_time_bins}')

    args.batch_size = 1
    # otherwise the collate function will through an error. 
    # This is only used in combination of include_incomplete=True during testing
    # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load weights from a checkpoint
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    else:
        raise ValueError("Please provide a checkpoint file.")

    # test_loader_desc = tqdm.tqdm(test_loader, desc="Testing")
    # evaluate on the validation set and save the predictions into a csv file.
    with open(args.output_path, 'w', newline='') as csvfile:
        with torch.no_grad():
            csv_writer = csv.writer(csvfile, delimiter=',')
        # add column names 'row_id', 'x', 'y'
            csv_writer.writerow(['row_id', 'x', 'y'])
            row_id = 0
            for idx in range(len(test_data)):
                data, target, metadata = test_data[idx]
                # for batch_idx, (data, target_placeholder) in enumerate(test_loader_desc):
                data = torch.tensor(data).unsqueeze(dim=0).to(args.device)
                output = model(data)

                # Important!
                # cast the output back to the downsampled sensor space (80x60)
                output = output * torch.tensor((640*factor, 480*factor)).to(args.device)

                # for sample in range(target.shape[0]):
                if metadata["index"] == (1, 84) or metadata["index"] == (9, 100):
                    print("Slice {}: Removing useless sample".format(metadata["index"]))
                    continue
                elif metadata["index"][-1] == 0:
                    print("Slice {}: First sample. Full inference".format(metadata["index"]))
                    for frame_id in range(target.shape[0]):
                        row_to_write = output[0][frame_id].tolist()
                        # prepend the row_id
                        row_to_write.insert(0, row_id)
                        csv_writer.writerow(row_to_write)
                        row_id += 1
                else:
                    print("Slice {}: Only generate the final prediction".format(metadata["index"]))
                    row_to_write = output[0][-1].tolist()
                    row_to_write.insert(0, row_id)
                    csv_writer.writerow(row_to_write)
                    row_id += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", type=str, default='test_config', \
                        help="path to JSON configuration file")
    # load weights from a checkpoint
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--output_path", type=str, default='./submission.csv')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")

    # quant hyperparameters
    parser.add_argument("--shift_bit", type=int, default=32, help="shift bit")
    parser.add_argument("--bias_bit", type=int, default=32, help="bias bit")
    parser.add_argument('--conv1_bit', type=int, default=8)
    parser.add_argument('--fixBN_ratio', type=float, default=0.3, help="When to fix BN during quantization training")

    args = parser.parse_args()

    main(args)
