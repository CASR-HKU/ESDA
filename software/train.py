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
from utils.training_utils import train_epoch, validate_epoch, top_k_checkpoints
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
# print(torch.cuda.is_available())

def setup_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子



def train(model, train_loader, val_loader, criterion, optimizer, args):
    best_val_loss = float("inf")
    best_val_p10_acc, best_val_p5_acc = 0, 0
    recorder = MetricRecorder(os.path.join(args.mlflow_path, "log.txt"))

    # Training loop
    for epoch in range(args.num_epochs):
        update_epoch(model, args, epoch)

        if args.train_quant and use_quant:
            unfreeze_model(model)

        lr = optimizer.optimizer.param_groups[0]['lr']
        print(f"Learning rate at epoch {epoch+1}/{args.num_epochs}: {lr}")
        model, train_loss, metrics = train_epoch(model, train_loader, criterion, optimizer, args, epoch)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metrics(metrics['tr_p_acc'], step=epoch)
        mlflow.log_metrics(metrics['tr_p_error'], step=epoch)
        optimizer.schedule_step(epoch)

        if args.train_quant and use_quant:
            freeze_model(model)

        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, args, epoch)
        recorder.update(recorder.transform_metric(train_loss, val_loss, metrics, val_metrics), epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save the new best model to MLflow artifact with 3 decimal places of validation loss in the file name
            torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), "model_best_val_loss.pth"))

            # DANGER Zone, this will delete files (checkpoints) in MLflow artifact
            # top_k_checkpoints(args, mlflow.get_artifact_uri())

        if best_val_p10_acc < val_metrics['val_p_acc']['val_p10_acc']:
            best_val_p10_acc = val_metrics['val_p_acc']['val_p10_acc']
            torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), "model_best_p10_acc.pth"))

        if best_val_p5_acc < val_metrics['val_p_acc']['val_p5_acc']:
            best_val_p5_acc = val_metrics['val_p_acc']['val_p5_acc']
            torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), "model_best_p5_acc.pth"))

        print(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metrics(val_metrics['val_p_acc'], step=epoch)
        mlflow.log_metrics(val_metrics['val_p_error'], step=epoch)
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f}")

    best_metrics = recorder.get_best()

    return model, best_metrics


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

    os.makedirs("/".join(args.mlflow_path.split('/')[:-1]), exist_ok=True)
    exp_record = ExpRecorder(os.path.join("/".join(args.mlflow_path.split('/')[:-1]), "exp_results.csv"))

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
        if not args.evaluate:
            shutil.copy(args.config_file, os.path.join(args.mlflow_path, "cfg.json"))
        process_commands(args)

        # Define your model, optimizer, and criterion
        model = eval(args.architecture)(args).to(args.device)
        use_upsample = model.use_heatmap

        args.train_quant = True if "Quant" in args.architecture and not args.evaluate else False
        if not args.train_quant:
            if args.checkpoint:
                model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

        Optim = Optimizer(args, model.parameters())
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)

        if args.loss == "mse":
            criterion = nn.MSELoss()
        elif args.loss == "weighted_mse":
            criterion = weighted_MSELoss(weights=torch.tensor((args.sensor_width/args.sensor_height, 1)).to(args.device), \
                                         reduction='mean')
        else:
            raise ValueError("Invalid loss name")
        if use_upsample:
            criterion = nn.MSELoss()

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
        # post_slicer_transform = transforms.Compose([
        #     SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        #     EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
        #                             n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
        # ])

        # train_data = EyeCenterRegressionSliceDataset(args, train_data_orig, train_slicer, \
        #                 metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}", istrain=True, use_upsample=use_upsample)
        # val_data = EyeCenterRegressionSliceDataset(args, val_data_orig, val_slicer, \
        #                 metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}", istrain=False, use_upsample=use_upsample)


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
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        # Train your model

        if args.evaluate:
            validate_epoch(model, val_loader, criterion, args, 0)
            sys.exit(0)

        model, metrics = train(model, train_loader, val_loader, criterion, Optim, args)
        exp_record.update(args, [0, params], metrics)

        # Save your model for the last epoch
        torch.save(model.state_dict(), os.path.join(mlflow.get_artifact_uri(), f"model_last_epoch{args.num_epochs}.pth"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # training management arguments     
    parser.add_argument("--mlflow_path", type=str, help="path to MLflow tracking server")
    parser.add_argument("--experiment_name", type=str, help="name of the experiment")
    parser.add_argument("--run_name", type=str, help="name of the run")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to train the model")
    parser.add_argument("--load", type=str, default="")

    # a config file 
    parser.add_argument("--config_file", type=str, default=None, help="path to JSON configuration file")
    parser.add_argument("--remove_cached", action="store_true", help="remove cached dataset")

    # training hyperparameters
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--num_epochs", type=int, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--batch_size", type=int, default=20, help="number of workers")

    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate the model")
    parser.add_argument("--checkpoint", type=str, default="", help="path to the model checkpoint")
    parser.add_argument("--debug",  "-d", action="store_true", help="evaluate the model")
    parser.add_argument("--seed", type=int, default=20, help="random seed")

    # quant hyperparameters
    parser.add_argument("--shift_bit", type=int, default=32, help="shift bit")
    parser.add_argument("--bias_bit", type=int, default=32, help="bias bit")
    parser.add_argument('--conv1_bit', type=int, default=8)
    parser.add_argument('--fixBN_ratio', type=float, default=0.3, help="When to fix BN during quantization training")

    args = parser.parse_args()

    setup_seed(args.seed)
    main(args)
