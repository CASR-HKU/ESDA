import math
import os
import sys

import cv2
import numpy as np
import torch.nn as nn
import tqdm
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torch.utils.data.distributed
import argparse
from models_int.mink_mobilenetv2 import MobileNetV2ME
from models_int.HAWQ_mobilenetv2 import Q_MobileNetV2

try:
    use_tb = True
    from torch.utils.tensorboard import SummaryWriter
except:
    use_tb = False
# from dataset.dataset import Dataset
import yaml
from dataset.loader import Loader
import utils.utils as utils

from config.settings import Settings
import MinkowskiEngine as ME
from utils.loss import Loss


# torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--int_dir', type=str, default='int')

    parser.add_argument('--data_path', help='Path to dataset', default="")
    parser.add_argument('--bias_bit', type=int, default=32)

    parser.add_argument('--dataset_percentage', '-dp', type=float, default=1)
    parser.add_argument('--conv1_level', type=int, default=8)
    parser.add_argument('--shift_bit', type=int, default=31)
    parser.add_argument('--load', type=str, default='', help='load model path')
    parser.add_argument('--ana_file', default=None, help='Path for error analyser')

    parser.add_argument('--drop_area_ratio', type=float, default=0.1)
    parser.add_argument('--drop_event_ratio', type=float, default=0.1)
    parser.add_argument('--slicing_time_window', type=float, default=500000)

    args = parser.parse_args()

    settings_filepath = args.settings_file
    settings = Settings(settings_filepath, generate_log=False)
    settings = utils.reinit_settings(settings, args)

    int_folder = args.int_dir
    os.makedirs(int_folder, exist_ok=True)

    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    if args.data_path != "":
        cfg["dataset"]["dataset_path"] = args.data_path

    nr_input_channels = utils.get_input_channel(settings.event_representation)
    num_workers = 0 if args.evaluate else settings.num_cpu_workers

    Dataset = utils.select_dataset(cfg["dataset"]["name"])
    MNIST = True if "MNIST" in cfg["dataset"]["name"] or "Poker" in cfg["dataset"]["name"] else False

    val_dataset = Dataset(cfg, mode="validation", shuffle=False, dataset_percentage=args.dataset_percentage,)
    if cfg["dataset"]["name"] == "Roshambo":
        nr_input_channels = 1

    nr_classes = val_dataset.nr_classes
    val_loader = Loader(val_dataset, batch_size=1, device=settings.gpu_device, num_workers=num_workers,
                        pin_memory=False, shuffle=False)

    train_quant, base_model = False, False
    if 'mink_mobilenetv2' in settings.model_name:
        baseline_model = MobileNetV2ME(num_classes=nr_classes, in_channels=nr_input_channels,
                                       width_mult=settings.width_mult,
                                       MNIST=MNIST, remove_depth=settings.remove_depth,
                                       drop_config=settings.drop_config,
                                       relu=settings.relu_type, model_type=settings.model_type)
        if settings.model_name == 'mink_mobilenetv2_quant':
            assert args.load, "Please specify the path to the baseline model"
            model = utils.load_model(args.load, baseline_model)
            model = Q_MobileNetV2(model, nr_input_channels, settings.width_mult, nr_classes, conv1_bit=args.conv1_level,
                                  fix_BN_threshold=args.fixBN_ratio, MNIST=MNIST, shift_bit=args.shift_bit,
                                  drop_config=settings.drop_config, model_type=settings.model_type,
                                  bias_bit=args.bias_bit)
    else:
        raise NotImplementedError

    param = sum(p.numel() for p in model.parameters())
    print("Number of ALL parameters: {}".format(param))
    param_noBN = sum(p.numel() for p in model.parameters() if len(p.shape) > 1)
    print("Number of parameters except BN: {}".format(param_noBN))
    model.to(settings.gpu_device)
    model = model.cuda()
    baseline_model.param = sum(p.numel() for p in baseline_model.parameters() if len(p.shape) > 1)  # param_noBN
    baseline_model = baseline_model.cuda()

    criterion = Loss()
    optimizer = None

    start_epoch, best_acc1 = utils.load_checkpoint(args, model, optimizer)
    cudnn.benchmark = True


    if args.evaluate:
        gen_npy(val_loader, model, criterion, settings.gpu_device, error_ana_path=args.ana_file,
                generate_int_model=args.generate_int_model, base_model=base_model,
                int_folder=int_folder)
        gen_json(val_loader, baseline_model, settings.gpu_device, base_model, int_folder)
        return

def gen_npy(val_loader, model, criterion, device, epoch=-1, writer=None, epoch_log=None, error_ana_path=None,
            generate_int_model=False, use_abs_drop=False, msg_log=None, base_model=False, int_folder=""):
    val_loader_desc = tqdm.tqdm(val_loader, ascii=True, mininterval=5, total=len(val_loader))
    model = model.eval()

    for i_batch, sample_batched in enumerate(val_loader_desc):

        if int_folder:
            np.save(os.path.join(int_folder, "input_Coordinates.npy"), sample_batched["coordinates"])
            np.save(os.path.join(int_folder, "input_Features.npy"), sample_batched["features"])

        with torch.no_grad():
            if base_model:
                model_output = model(sample_batched["histograms"].to(device))
            else:
                minknet_input = ME.SparseTensor(
                    coordinates=sample_batched["coordinates"], features=sample_batched["features"], device=device
                )
                model_output = model(minknet_input, int_folder)

        return


def gen_json(val_loader, model, device, base_model=False, int_folder=""):
    val_loader_desc = tqdm.tqdm(val_loader, ascii=True, mininterval=5, total=len(val_loader))
    model = model.eval()

    dataset_name = val_loader.loader.dataset.dataset.name
    size = val_loader.loader.dataset.transform.interplote_size.tolist()
    model.json_file = os.path.join(int_folder, "model.json")

    for i_batch, sample_batched in enumerate(val_loader_desc):

        with torch.no_grad():
            if base_model:
                model_output = model(sample_batched["histograms"].to(device))
            else:
                minknet_input = ME.SparseTensor(
                    coordinates=sample_batched["coordinates"], features=sample_batched["features"], device=device
                )
                model_output = model(minknet_input, dataset_name, size)
    return




if __name__ == '__main__':
    main()
