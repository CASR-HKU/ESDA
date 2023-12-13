import random

import numpy as np
import cv2
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import argparse
import yaml
import utils.utils as utils
from settings import Settings

# target_cls_idx = [77, 74, 78, 28]
# ASL candidate

# NMNIST candidate
# sample_sizes = [1024, 2048, 4096, 10000]
# denoises = [10000, 30000, 50000, 100000]

def main():
    parser = argparse.ArgumentParser(description='Preprocess & Visualization')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--val', action='store_true', help='Load validation set; Default training')

    # Preprocess options
    # parser.add_argument('--preprocess', action='store_true', help='Whether conduct preprocess')
    # parser.add_argument('--save_dir', "-s", type=str, default='')
    # parser.add_argument('--vis_hist', action='store_true', help='Visualize histogram')
    # parser.add_argument('--window_size', type=float, default=0.25)
    # parser.add_argument('--overlap_ratio', type=float, default=0)
    # parser.add_argument('--denoise_time', type=float, default=-1)
    # parser.add_argument('--use_denoise', action='store_true', help='Whether using denoise events in preprocessed data')

    args = parser.parse_args()

    settings_filepath = args.settings_file

    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)
    Dataset = utils.select_dataset(cfg["dataset"]["name"])
    mode = "validation" if args.val else "training"

    cfg["transform"]["flip"]["p"] = 0
    cfg["transform"]["shift"]["max_shift"] = 0

    if "ASL" in cfg["dataset"]["name"]:
        sample_sizes = [1024]
        denoises = [-1]
    elif "NMNIST" in cfg["dataset"]["name"]:
        sample_sizes = [1024]
        denoises = [-1]
    elif "CIFAR" in cfg["dataset"]["name"]:
        sample_sizes = [-1]
        denoises = [-1]
    elif "DVS" in cfg["dataset"]["name"]:
        sample_sizes = [2048]
        denoises = [-1]
    elif "Cal" in cfg["dataset"]["name"]:
        sample_sizes = [5000]
        denoises = [-1]
    elif "Poker" in cfg["dataset"]["name"]:
        sample_sizes = [256]
        denoises = [-1]
    else:
        raise NotImplementedError
    # dataset_raw = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
    # dataset_denoise = Dataset(cfg, mode=mode, shuffle=False, use_denoise=True)
    #
    # cfg["transform"]["sample"]["window"] = 50000
    #
    # dataset_raw_sampled = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
    # dataset_denoise_sampled = Dataset(cfg, mode=mode, shuffle=False, use_denoise=True)
    # datasets = [dataset_raw, dataset_denoise, dataset_raw_sampled, dataset_denoise_sampled]
    # names = ["sample4096", "sample10000", "sample30000", "sample50000"]

    datasets, names = [], []
    for sample_size in sample_sizes:
        for denoise in denoises:
            cfg["transform"]["sample"]["window"] = sample_size
            cfg["transform"]["denoise"]["time"] = denoise
            dataset = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
            datasets.append(dataset)
            names.append("sample{}-denoise{}k".format(sample_size, int(denoise/1000)))
    # cfg["transform"]["sample"]["window"] = 4096
    # dataset1 = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
    # cfg["transform"]["sample"]["window"] = 4096
    # dataset2 = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
    # cfg["transform"]["sample"]["window"] = 10000
    # dataset3 = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)
    # cfg["transform"]["sample"]["window"] = 30000
    # dataset4 = Dataset(cfg, mode=mode, shuffle=False, use_denoise=False)

    # datasets = [dataset1, dataset2, dataset3, dataset4]
    # names = ["sample_1k", "sample_4096", "sample_10k", "sample_30k"]

    model_input_size = torch.tensor((175, 175))

    while True:
        idx = random.randint(0, len(datasets[0]) - 1)
        samples = [d[idx] for d in datasets]
        # if target_cls_idx:
        #     if samples[0][1] not in target_cls_idx:
        #         continue
        histograms = [sample["histogram"].unsqueeze(dim=0) for sample in samples]
        imgs = []
        for i, histogram in enumerate(histograms):
            imgs.append(utils.histogram_visualize(histogram, model_input_size, [names[i]]))
        merged_img = utils.merge_vis(imgs, (len(denoises), len(sample_sizes)))
        tmp_img = np.full((30, merged_img.shape[1], 3), 128, dtype="uint8")
        # cv2.putText(tmp_img, "Dataset: {}, Label: {}, Event length: {}, Time {}".format(
        #     cfg["dataset"]["name"], datasets[0].dataset.object_classes[samples[0]["label"]], len(samples[0]["event"]),
        #     samples[0]["event"][-1][2]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(tmp_img,  "Label: {}".format(
            datasets[0].dataset.object_classes[samples[0]["label"]]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        # print("Current label is ")
        image = np.vstack((tmp_img, merged_img))
        cv2.imshow("show", image)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
