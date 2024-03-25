import math
import numpy as np
import torch
import cv2
from dataset import SliceLongEventsToShort, EventSlicesToVoxelGrid
import tonic.transforms as transforms
from tonic.transforms import ToTimesurface, ToImage
import os


def generate_post_transform(args):
    transform = [SliceLongEventsToShort(time_window=int(10000 / args.temporal_subsample_factor), overlap=0,
                                        include_incomplete=True)]
    cached_item = ""
    if "represent2d" not in args:
        transform.append(EventSlicesToVoxelGrid(sensor_size=(int(640 * args.spatial_factor), int(480 * args.spatial_factor), 2), \
                               n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization))
    else:
        if args.represent2d == "voxels":
            transform.append(
                EventSlicesToVoxelGrid(sensor_size=(int(640 * args.spatial_factor), int(480 * args.spatial_factor), 2), \
                                       n_time_bins=args.n_time_bins,
                                       per_channel_normalize=args.voxel_grid_ch_normaization))
        elif args.represent2d == "time_surface":
            cached_item = "_ts"
            transform.append(
                ToTimesurface(sensor_size=(int(640 * args.spatial_factor), int(480 * args.spatial_factor), 2), \
                              dt=0, tau=0))
        elif args.represent2d == "image":
            cached_item = "_image"
            transform.append(
                ToImage(sensor_size=(int(640 * args.spatial_factor), int(480 * args.spatial_factor), 2))
            )
        else:
            raise NotImplementedError("Not supported")

    return transforms.Compose(transform), cached_item


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"


def process_commands(args):
    import mlflow, sys
    train_cmd = generate_cmd(sys.argv[1:])
    # train_component = generate_cmd(sys.argv[3:])[:-1]
    cmd_file = os.path.join(args.mlflow_path, "cmd.txt")
    eval_cmd = "python train.py -e {} --config_file={} --checkpoint {}\n".format(train_cmd[:-1],
        os.path.join(args.mlflow_path, "cfg.json"), os.path.join(mlflow.get_artifact_uri(), "model_best_p5_acc.pth")
    )
    test_cmd = "python test.py {} --config_file={} --checkpoint {} --output_path {}\n".format(train_cmd[:-1],
        os.path.join(args.mlflow_path, "cfg.json"), os.path.join(mlflow.get_artifact_uri(), "model_best_p5_acc.pth"),
        os.path.join(args.mlflow_path, "submission.csv")
    )
    error_cmd = "python error_analysis.py {} --config_file={} --checkpoint {} --output_folder {}\n".format(train_cmd[:-1],
        os.path.join(args.mlflow_path, "cfg.json"), os.path.join(mlflow.get_artifact_uri(), "model_best_p5_acc.pth"),
        os.path.join(args.mlflow_path, "error_analysis")
    )
    cmds = [train_cmd, eval_cmd, test_cmd, error_cmd]
    with open(cmd_file, "w") as f:
        f.writelines(cmds)




def update_epoch(model, args, epoch):
    for module in model.modules():
        if hasattr(module, "epoch_counter"):
            module.epoch_counter = epoch / args.num_epochs
        if hasattr(module, "epoch_ratio"):
            module.epoch_ratio = (epoch+1) / args.num_epochs


def process(coord, height, width):
    coord[:, 0] = coord[:, 0] * width
    coord[:, 1] = coord[:, 1] * height
    coord = coord.astype(np.int32)
    return coord


def draw_points(images, coords, color=(0, 255, 0)):
    assert len(images) == len(coords)
    for idx in range(len(images)):
        images[idx] = cv2.circle(images[idx], tuple(coords[idx]), 2, color, 2)
    return images


def draw_distance(images, distances, color=(0, 255, 0)):
    assert len(images) == len(distances)
    for idx in range(len(images)):
        images[idx] = cv2.putText(images[idx], f"{distances[idx]:.2f}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.3, color, 2)
    return images


def merge(frames_list, h, w):

    def get_output_hw(number):
        row = int(math.sqrt(number))
        col = math.ceil(number / row)
        padding = int(row * col - number)
        return row, col, padding

    row, col, padding = get_output_hw(len(frames_list))
    black_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(padding):
        frames_list.append(black_img)
    row_images = []
    for i in range(0, len(frames_list), col):
        row_img = np.concatenate(frames_list[i:i+col], axis=1)
        row_images.append(row_img)
    out_img = np.concatenate(row_images, axis=0)
    return out_img

def merge_images(frames_list, sizes):

    def get_output_hw(number):
        row = int(math.sqrt(number))
        col = math.ceil(number / row)
        padding = int(row * col - number)
        return row, col, padding

    row, col, padding = get_output_hw(len(frames_list))

    black_img = np.zeros((sizes[0], sizes[1], 3), dtype=np.uint8)
    for i in range(padding):
        frames_list.append(black_img)
    row_images = []
    for i in range(0, len(frames_list), col):
        row_img = np.concatenate(frames_list[i: i + col], axis=1)
        row_images.append(row_img)
    out_img = np.concatenate(row_images, axis=0)
    return out_img


def load_pretrain(args, model):
    if not args.load:
        return model
    device = args.device
    try:
        checkpoint_dict = torch.load(args.load, map_location=device)['state_dict']
    except:
        checkpoint_dict = torch.load(args.load, map_location=device)

    model_dict = model.state_dict()
    # update_dict = {k: v for k, v in model_dict.items() if k in checkpoint_dict.keys()}
    update_keys = [k for k, v in model_dict.items() if k in checkpoint_dict.keys()]
    update_dict = {k: v for k, v in checkpoint_dict.items() if k in update_keys}
    # update_dict = load_components(args.load_components, update_dict)
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model
