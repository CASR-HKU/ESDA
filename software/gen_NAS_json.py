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
    configs_folder = args.config_folder
    config_files = os.listdir(configs_folder)

    os.makedirs(args.output_folder, exist_ok=True)

    sample_config_file = os.path.join(configs_folder, config_files[0])
    with open(sample_config_file, 'r') as f:
        config = json.load(f)
    # Overwrite hyperparameters with command-line arguments
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    args = argparse.Namespace(**config)


    factor = args.spatial_factor # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor # downsampling original 100Hz label to 20Hz

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
    train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

    val_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                    seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

    post_slicer_transform, cached_item = generate_post_transform(args)
    val_data = SlicedDataset(val_data_orig, val_slicer,
                             transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}" + cached_item)

    val_cached_path = f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}' + cached_item

    val_data = DiskCachedDataset(val_data, cache_path=val_cached_path)
    augment_path = None
    if "augment_path" in args:
        augment_path = args.augment_path

    val_data = EyeCenterRegressionDataset(val_data, val_cached_path, augment_path=augment_path, use_upsample=False)

    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

    for config_file in config_files:
        model = MobileNetSubmanifold(args).to(args.device)

        params = print_model_param_nums(model)
        print("Parameters of current model is {}".format(params))
        model.param = params

        model = model.eval()
        dataset_name = "EyeTracking"
        size = (args.sensor_height*args.spatial_factor, args.sensor_width*args.spatial_factor)
        model.json_file = os.path.join(args.output_folder, config_file)

        for i_batch, (inputs, target, _) in enumerate(val_loader):
            with torch.no_grad():
                model(inputs, dataset_name, size)
                # return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train the model")
    parser.add_argument("--config_folder", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument('--output_folder', type=str, default="", help="json folder")

    args = parser.parse_args()
    main(args)
