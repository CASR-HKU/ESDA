
import numpy as np
import torch
import os
import argparse
import yaml
import utils.utils as utils


def main():
    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--output_file', default="npy/NCal", help='Path folder')
    parser.add_argument('--data_path', default="", help='Path to dataset')

    args = parser.parse_args()
    os.makedirs(args.output_file, exist_ok=True)

    settings_filepath = args.settings_file

    with open(settings_filepath, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    if args.data_path != "":
        cfg["dataset"]["dataset_path"] = args.data_path
    Dataset = utils.select_dataset(cfg["dataset"]["name"])
    val_dataset = Dataset(cfg, mode="validation", cross_valid=-1, shuffle=False)

    height, width = cfg["transform"]["height"], cfg["transform"]["width"]

    for idx in range(len(val_dataset)):
        data = val_dataset[idx]
        print(data)
        mask = torch.zeros(height, width)
        for c in data['coordinates']:
            mask[c[0], c[1]] = 1
        np.save(os.path.join(args.output_file, "{}.npy".format(idx)), mask.numpy())


if __name__ == '__main__':
    main()
