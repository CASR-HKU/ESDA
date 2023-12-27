import os
import torch
import sys
import numpy as np
import math, cv2


def get_input_channel(event):
    if event == 'histogram':
        nr_input_channels = 2
    elif event == 'event_queue':
        nr_input_channels = 30
    elif event == "time_surface":
        nr_input_channels = 1
    elif event == "least_timestamp":
        nr_input_channels = 2
    else:
        raise NotImplementedError
    return nr_input_channels


def check_abs_sum_drop(settings):
    use_abs_drop = False
    if settings.drop_config and "abs_sum" in settings.drop_config["type"]:
        use_abs_drop = True
    return use_abs_drop


def multiple_validate(val_func, val_loader, model, criterion, device, epoch=-1, writer=None, epoch_log=None, ana_file=None,
              msg_log=None, times=5, base_model=False):
    metrics = []
    for t in range(times):
        msg_log = None if t > 0 else msg_log
        metric, matrix = val_func(val_loader, model, criterion, device, error_ana_path=ana_file, epoch=epoch,
                                  writer=writer, epoch_log=epoch_log, msg_log=msg_log, base_model=base_model)
        metrics.append(metric)

    metrics = np.array(metrics).mean(axis=0).tolist()
    return metrics, matrix


def check_random(settings, cfg):
    random = False
    if "sample_type" in cfg["transform"]["sample"]:
        if cfg["transform"]["sample"]["sample_type"] == "random":
            random = True
    print("Using random: Validation average for 5 times") if random else print("Not using random")
    return random


def calculate_weight_sparsity(model, file="weight_sparsity.txt"):
    with open(file, "w") as f:
        for n, w in model.state_dict().items():
            if "weight" in n:
                sparsity = (w == 0).sum() / w.numel()
                print(f"{n}: {sparsity}", file=f)


def update_epoch(model, args, epoch):
    for module in model.modules():
        if hasattr(module, "epoch_counter"):
            module.epoch_counter = epoch / args.epochs
        if hasattr(module, "epoch_ratio"):
            module.epoch_ratio = (epoch+1) / args.epochs


def load_model(path, model, device="cuda:0"):
    try:
        checkpoint_dict = torch.load(path, map_location=device)['state_dict']
    except:
        checkpoint_dict = torch.load(path, map_location=device)

    model_dict = model.state_dict()
    # update_dict = {k: v for k, v in model_dict.items() if k in checkpoint_dict.keys()}
    update_keys = [k for k, v in model_dict.items() if k in checkpoint_dict.keys()]
    update_dict = {k: v for k, v in checkpoint_dict.items() if k in update_keys}
    # update_dict = load_components(args.load_components, update_dict)
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    return model

def reinit_settings(settings, args):
    if settings.steps_lr < 5:
        settings.steps_lr = int(args.epochs/(settings.steps_lr+1))
    return settings

def select_dataset(name):
    if "preprocess" not in name:
        from dataset.dataset import Dataset
        return Dataset
    else:
        from dataset.dataset import PreprocessedDataset
        return PreprocessedDataset


def histogram_visualize(histogram, model_input_size, labels):
    histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(model_input_size))
    histogram = histogram.permute(0, 2, 3, 1)

    locations, features = denseToSparse(histogram)
    return vis(locations, features, model_input_size, labels)

def vis(locations, features, model_input_size, labels, save_num=9):
    try:
        import visualizations
    except:
        import utils.visualizations as visualizations
    save_num = min(len(labels), save_num)
    h = int(math.sqrt(save_num))
    w = int(save_num / h)
    images = []
    for idx in range(save_num):
        batch_one_mask = locations[:, -1] == idx
        vis_locations = locations[batch_one_mask, :2]
        feature = features[batch_one_mask, :]
        image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), model_input_size,
                                                  features=feature.cpu().numpy())
        if labels is not None:
            tmp_img = np.full((30, image.shape[1], 3), 128, dtype="uint8")
            cv2.putText(tmp_img, labels[idx], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 255), 2)
            image = np.vstack((image, tmp_img))
            block_img = np.full((image.shape[0], 20, 3), 128, dtype="uint8")
            image = np.hstack((image, block_img))

        images.append(image)
    return merge_vis(images, (h, w))

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
    locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense_tensor[select_indices], dim=-2)

    return locations, features

def merge_vis(image_ls, size):
    h, w = size
    tmp_images = []
    for idx in range(h):
        tmp_images.append(np.concatenate(image_ls[idx*h: idx *h+ w], axis=1))
    return np.concatenate(tmp_images, axis=0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def tp2str(tp, rounds=None):
    if rounds is None:
        rounds = [4 for _ in range(len(tp))]

    string = ""
    for item, r in zip(tp, rounds):
        string += str(round(float(item), r))
        string += "\t"
    return string + "\n"


def float2strList(ls, r=4):
    return [str(round(item, r)) for item in ls]


def load_components(comps, model_dicts):
    if "all" in comps:
        return model_dicts
    updated_dict = {}
    for comp in comps:
        for k,v in model_dicts.items():
            if comp in k:
                updated_dict[k] = v
    return updated_dict


def load_checkpoint(args, model, optimizer, device="cuda:0"):
    best_acc = 0
    global iteration
    start_epoch = -1
    if args.auto_resume and not args.evaluate:
        assert args.save_dir, "Please specify the auto resuming folder"
        resume_path = os.path.join(args.save_dir, "ckpt.pth.tar")
        if os.path.isfile(resume_path):
            print(f"=> loading checkpoint '{resume_path}'")
            checkpoint = torch.load(resume_path)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{resume_path}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_acc']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            # print('check', checkpoint)
            start_epoch = checkpoint['epoch']
            try:
                iteration = checkpoint['iteration']
            except:
                iteration = args.batch_size * start_epoch
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['best_acc']
            # print(f"=> loaded checkpoint '{args.resume}'' (epoch {checkpoint['epoch']}, best prec1 {checkpoint['best_prec1']})")
        else:
            msg = "=> no checkpoint found at '{}'".format(args.resume)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)
    elif args.load:
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
    return start_epoch, best_acc


def load_int_ckpt(model, path):
    import torch
    checkpoint_dict = torch.load(path, map_location="cuda:0")
    for k, v in checkpoint_dict.items():
        try:
            if "." in k:
                blocks, b_idx, layer_names = k.split(".")
                if blocks == "blocks":
                    model.blocks[int(b_idx)].register_buffer(layer_names, v)
                else:
                    continue
            else:
                model.register_buffer(k, v)
        except:
            pass

def generate_test_cmd(string, args):
    string += " -e --load {} --settings_file={}".format(os.path.join(args.save_dir, "ckpt.best.pth.tar"),
                                                        os.path.join(args.save_dir, "settings.yaml"))
    return string


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"


def handle_logger(args):
    cmd = generate_cmd(sys.argv[1:])
    config_log, train_log, test_log, msg_log = os.path.join(args.save_dir, "config.log"), \
        os.path.join(args.save_dir, "train.log"), \
        os.path.join(args.save_dir, "test.log"), \
        os.path.join(args.save_dir, "message.log"),
    test_cmd = generate_test_cmd(cmd[:-1], args)
    if not (os.path.exists(config_log) and not args.auto_resume):
        with open(config_log, "w") as f:
            f.write(cmd + "\n\n")
            f.write(test_cmd + "\n\n")
            print('Args:', args, file=f)
            f.write("\n")
            for k, v in vars(args).items():
                f.write("{k} : {v}\n".format(k=k, v=v))
            f.flush()
            os.fsync(f)


def set_gumbel(intervals, temps, epoch_ratio, remove_gumbel):
    assert len(intervals) == len(temps), "Please reset your gumbel"
    len_gumbel = len(intervals)
    gumbel_temp = temps[-1]
    for idx in range(len(intervals)):
        if intervals[len_gumbel-idx-1] > epoch_ratio:
            gumbel_temp = temps[len_gumbel-idx-1]
        else:
            break
    gumbel_noise = False if epoch_ratio > remove_gumbel else True
    return gumbel_temp, gumbel_noise


def sort_paths(paths):
    sorted_paths = []
    for im_idx in range(len(paths[0])):
        for b_idx in range(len(paths)):
            sorted_paths.append(paths[b_idx][im_idx])
    return sorted_paths


def save_pred_and_confidence(outputs, target, file=None):
    confs, preds = [], []
    for output in outputs:
        conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        confs.append(conf.cpu().tolist())
        preds.append((pred == target).int().cpu().tolist())

    if file is None:
        file = "jntm.npz"
    np.savez(file, pred=torch.tensor(preds), conf=torch.tensor(confs))
    a = 1
    # pred_file["pred"] = torch.tensor(preds)
    # pred_file["conf"] = torch.tensor(confs)
    # pred_file.close()

import shutil
def save_checkpoint(state, filename, is_best, progress):
    save_progress = [0.8]
    print("Saving checkpoint: {}".format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
    if progress in save_progress:
        shutil.copyfile(filename, filename.replace('pth.tar', '{}.pth.tar'.format(progress)))


def input_order(num_segments):
    S = [np.floor((num_segments - 1) / 2), 0, num_segments - 1]
    q = 2
    while len(S) < num_segments:
        interval = np.floor(np.linspace(0, num_segments - 1, q + 1))
        for i in range(0, len(interval) - 1):
            a = interval[i]
            b = interval[i + 1]
            ind = np.floor((a + b) / 2)
            if not ind in S:
                S.append(ind)
        q *= 2
    S = [int(s) for s in S]
    print('Input Order:', S)
    return S


def adjust_learning_rate(args, optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def fill_full_target(target, local_rank=-1):
    tmp = torch.tensor(-1).expand((target.shape[0], 2)).cuda() if local_rank == -1 else \
        torch.tensor(-1).expand((target.shape[0], 2)).to(local_rank)
    return torch.cat((target.unsqueeze(dim=1), tmp), dim=1)


def check_trained_folder(args):
    if args.evaluate:
        return True
    path, auto_resume, resume = args.save_dir, args.auto_resume, args.resume
    test_file = os.path.join(path, "test.log")
    if auto_resume or resume:
        return True
    if not os.path.exists(test_file):
        return True
    with open(test_file, "r") as f:
        lines = f.readlines()
    if len(lines) > 5:
        return False
    else:
        return True

