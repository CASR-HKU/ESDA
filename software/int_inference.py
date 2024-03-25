"""
This script implements a PyTorch deep learning training pipeline for an eye tracking application.
It includes a main function to pass in arguments, train and validation functions, and uses MLflow as the logging library.
The script also supports fine-grained deep learning hyperparameter tuning using argparse and JSON configuration files.
All hyperparameters are logged with MLflow.

Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""
import sys
import shutil
from utils.optimizer import Optimizer
import argparse, json, os, mlflow
import torch
import torch.nn as nn
try:
    from model.HAWQ_quant_module.quant_modules import freeze_model, unfreeze_model
    use_quant = True
except:
    use_quant = False
from torch.utils.data import DataLoader
from utils.metrics import weighted_MSELoss
from utils.training_utils import train_epoch, gen_npy, top_k_checkpoints, gen_json
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from utils.recorder import MetricRecorder
from dataset.regression_dataset import EyeCenterRegressionDataset, EyeCenterRegressionSliceDataset
from model import *
from utils.recorder import ExpRecorder
from model.benchmark import print_model_param_flops, print_model_param_nums
from utils.utils import load_pretrain, generate_post_transform, update_epoch, process_commands




def setup_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



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

    os.makedirs(args.int_folder, exist_ok=True)
    args.evaluate = True
    # Set up MLflow logging
    mlflow.set_tracking_uri(args.mlflow_path)
    mlflow.set_experiment(experiment_name=args.experiment_name)

    # Start MLflow run
    with mlflow.start_run(run_name=args.run_name):
        # dump this training file to MLflow artifact
        mlflow.log_artifact(__file__)

        # Log all hyperparameters to MLflow
        mlflow.log_params(vars(args))
        # also dump the args to a JSON file in MLflow artifact
        with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
            json.dump(vars(args), f)
        shutil.copy(args.config_file, os.path.join(args.mlflow_path, "cfg.json"))
        process_commands(args)

        # Define your model, optimizer, and criterion
        model = eval(args.architecture)(args).to(args.device)
        use_upsample = model.use_heatmap

        args.train_quant = True if "Quant" in args.architecture and not args.evaluate else False
        if not args.train_quant:
            if args.checkpoint:
                model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

        factor = args.spatial_factor # spatial downsample factor
        temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

        # First we define the label transformations
        label_transform = transforms.Compose([
            ScaleLabel(factor),
            TemporalSubsample(temp_subsample_factor),
            NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
        ])

        # flops = print_model_param_flops(model, device=args.device, input_width=int(640*factor),
        #                                 input_height=int(480*factor), temporal_length=args.train_length)
        # print("FLOPs of current model is {}".format(flops))
        params = print_model_param_nums(model)
        print("Parameters of current model is {}".format(params))

        # Then we define the raw event recording and label dataset, the raw events spatial coordinates are also downsampled
        train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", \
                        transform=transforms.Downsample(spatial_factor=factor), 
                        target_transform=label_transform)
        val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                        transform=transforms.Downsample(spatial_factor=factor),
                        target_transform=label_transform)

        # Then we slice the event recordings into sub-sequences. 
        # The time-window is determined by the sequence length (train_length, val_length) 
        # and the temporal subsample factor.
        slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
        train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

        train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                        seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
        # the validation set is sliced to non-overlapping sequences
        val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                        seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

        # After slicing the raw event recordings into sub-sequences, 
        # we make each subsequences into your favorite event representation, 
        # in this case event voxel-grid

        post_slicer_transform, cached_item = generate_post_transform(args)
        # We use the Tonic SlicedDataset class to handle the collation of the sub-sequences into batches.
        train_data = SlicedDataset(train_data_orig, train_slicer,
                                   transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}" + cached_item)
        val_data = SlicedDataset(val_data_orig, val_slicer,
                                 transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}" + cached_item)

        train_cached_path = f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}' + cached_item
        val_cached_path = f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}' + cached_item

        # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
        train_data = DiskCachedDataset(train_data, cache_path=train_cached_path)
        val_data = DiskCachedDataset(val_data, cache_path=val_cached_path)
        augment_path = None
        if "augment_path" in args:
            augment_path = args.augment_path
        train_data = EyeCenterRegressionDataset(train_data, train_cached_path, augment_path=augment_path, istrain=True,
                                                use_upsample=use_upsample)
        val_data = EyeCenterRegressionDataset(val_data, val_cached_path, augment_path=augment_path,
                                              use_upsample=use_upsample)

        # Finally we wrap the dataset with pytorch dataloader
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

        # Train your model
        gen_npy(model, val_loader, "", args, 0)
        base_model = eval(args.architecture[:-5])(args).to(args.device)
        base_model.param = params
        gen_json(base_model, val_loader, args, int_folder=args.int_folder)
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    parser.add_argument("--mlflow_path", type=str, help="path to MLflow tracking server")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train the model")

    # a config file 
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument("--remove_cached", action="store_true", help="remove cached dataset")

    # training hyperparameters
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")

    parser.add_argument("--checkpoint", type=str, default="", help="path to the model checkpoint")
    parser.add_argument("--debug",  "-d", action="store_true", help="evaluate the model")

    # quant hyperparameters
    parser.add_argument("--shift_bit", type=int, default=32, help="shift bit")
    parser.add_argument("--bias_bit", type=int, default=32, help="bias bit")
    parser.add_argument('--conv1_bit', type=int, default=8)
    parser.add_argument('--fixBN_ratio', type=float, default=0.3, help="When to fix BN during quantization training")
    parser.add_argument('--int_folder', type=str, default="", help="integer folder")

    args = parser.parse_args()
    main(args)
