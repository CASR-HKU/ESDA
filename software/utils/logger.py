import torch
import utils.utils as utils
import os
import sys
import numpy as np


class PruneRatioLogger:
    def __init__(self):
        self.prune_ratio = []

    def update(self, model):
        self.threshold = [m.threshold.tolist() for n, m in model.named_modules() if "prune" in n and ".prune." not in n]
        self.prune_ratio.append([m.skip_ratio for n, m in model.named_modules() if "prune" in n and ".prune." not in n])
        # try:
        #     self.threshold.append([m.threshold for n, m in model.named_modules() if "prune" in n and "mink" not in n])
        # except:
        #     self.threshold.append([m.threshold.cpu() for n, m in model.named_modules() if "prune" in n and "mink" not in n])

    def release(self, file=None, epoch=-1):
        prune_ratio = np.array(self.prune_ratio).mean(axis=0)
        print("Threshold: ", self.threshold)
        print("Prune ratio: ", prune_ratio)
        if file is not None:
            file.write("Epoch {}\n".format(epoch))
            print("Prune ratio: ", prune_ratio, file=file)
            print("Threshold: ", self.threshold, file=file)
            file.write("\n")
        return self.prune_ratio, self.threshold

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MaskTransformer:
    def __init__(self):
        pass

    def update(self, mask1, mask2, mask3, mask4):
        self.mask1 = mask1
        self.mask2 = mask2
        self.mask3 = mask3
        self.mask4 = mask4

    def update_dynconv(self, masks):
        self.masks = masks

    def form_mask(self):
        mask_dict_ls = []
        for m in [self.mask1, self.mask2, self.mask3, self.mask4]:
            mask_dict_ls.append({"std": m[0], "dilate": m[1]})
        return mask_dict_ls

    def form_to_viz(self):
        masks = [self.mask1[0], self.mask2[0], self.mask3[0], self.mask4[0]]
        reorganized_masks = []
        for m_idx in range(len(masks[0])):
            tmp = []
            for s_idx in range(len(masks)):
                tmp.append(masks[s_idx][m_idx])
            reorganized_masks.append(tmp)
        return reorganized_masks

    def to_dynconv(self):
        processed_masks = []
        for idx, masks in enumerate(self.masks):
            processed_masks.append({"dilate": masks["dilate"].hard.int(), "std": masks["std"].hard.int()})
        return processed_masks


class SpatialRecorder:
    def __init__(self, path):
        self.path = path
        self.records = []
        self.samples = []
        self.cnt = 0

    def update(self, masks, names=None):
        if not self.records:
            self.mask_len = len(masks)
            self.records = [(masks[idx]["dilate"].hard.sum((1, 2, 3)) / masks[idx]["dilate"].hard[0].numel()).tolist()
                            for idx in range(len(masks))]
        else:
            for record, mask in zip(self.records, masks):
                record += (mask["dilate"].hard.sum((1, 2, 3)) / mask["dilate"].hard[0].numel()).tolist()

        if names is None:
            names = [idx+self.cnt for idx in range(masks[0]["dilate"].hard.shape[0])]
        else:
            names = list(map(lambda x: x.split("/")[-1], names))
        self.cnt += masks[0]["dilate"].hard.shape[0]
        self.samples = names if not self.samples else self.samples + names

    def summary(self):
        with open(self.path, "w") as f:
            content = [self.samples] + self.records
            f.write("Sample " + " ".join(list(map(lambda x: "layer_{}".format(x), range(self.mask_len)))) + "\n")
            for sample in range(len(content[0])):
                for layer in range(len(content)):
                    f.write(str(content[layer][sample]) + "\t")
                f.write("\n")


def log_global(metrics, args, global_log):
    if global_log is not None:
        with open(global_log, "a+") as f:
            rounds = [2, 2, 2, 4, 4, 4, 2]
            f.write(args.save_dir.split("/")[-1] + "\t")
            f.write(utils.tp2str(metrics, rounds))


def get_metrics(num):
    return [AverageMeter() for _ in range(num)]


def layer_count(args):
    if args.model == "resnet50":
        layer_cnt = 16
    elif args.model == "resnet101":
        layer_cnt = 36
    elif args.model == "MobileNetV2":
        layer_cnt = 17
    elif args.model == "resnet32":
        layer_cnt = 15
    elif args.model == "MobileNetV2_32x32":
        layer_cnt = 12
    else:
        layer_cnt = 20
    return layer_cnt


def record_msg(msg_path, msgs, epoch=None):
    if msg_path is not None:
        if isinstance(msg_path, str):
            with open(msg_path, "a+") as f:
                if epoch is not None:
                    f.write("Epoch {}\n".format(epoch))
                for msg in msgs:
                    f.write(msg)
                f.flush()
                os.fsync(f)
        else:
            f = msg_path
            if epoch is not None:
                f.write("Epoch {}\n".format(epoch))
            for msg in msgs:
                f.write(msg)
            f.flush()
            os.fsync(f)


def open_with_title(file_path, title=""):
   if not os.path.exists(file_path):
       file = open(file_path, "w")
   else:
       file = open(file_path, "a+")
   if title:
       file.write(title+"\n")
   return file


def handle_loggers(args, model=None, global_file=True):
    train_log, test_log, msg_log, global_log = None, None, None, \
                                               os.path.join(os.path.join("/".join(args.save_dir.split("/")[:-1])), "group_result.txt")

    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))

        if global_file:
            if not os.path.exists(global_log):
                with open(global_log, "w") as f:
                    f.write("Name\ttop1\ttop5\tmAP\ttask-loss\tsparsity-loss\tloss\tMMac\n")
        else:
            global_log = None

        if model is not None and not os.path.exists(os.path.join(args.save_dir, "model.log")):
            with open(os.path.join(args.save_dir, "model.log"), "w") as f:
                print(model, file=f)

        config_log, train_log, test_log, msg_log = os.path.join(args.save_dir, "config.log"), \
                                                   os.path.join(args.save_dir, "train.log"), \
                                                   os.path.join(args.save_dir, "test.log"), \
                                                   os.path.join(args.save_dir, "message.log"),

        cmd = utils.generate_cmd(sys.argv[1:])
        test_cmd = utils.generate_test_cmd(cmd[:-1], args)
        if not (os.path.exists(os.path.join(args.save_dir, "ckpt.pth.tar")) and not args.auto_resume):
            with open(config_log, "w") as f:
                f.write(cmd + "\n\n")
                f.write(test_cmd + "\n\n")
                print('Args:', args, file=f)
                f.write("\n")
                for k, v in vars(args).items():
                    f.write("{k} : {v}\n".format(k=k, v=v))
                f.flush()
                os.fsync(f)

    elif args.evaluate:
        if global_file:
            try:
                if not os.path.exists(global_log):
                    with open(global_log, "w") as f:
                        f.write("Name\ttop1\tMMac\ttop5\ttask-loss\tsparsity-loss\n")
            except:
                global_log = None
        else:
            global_log = None
    if args.save_dir:
        os.makedirs(os.path.join(args.save_dir, "analysis"), exist_ok=True)
    return train_log, test_log, msg_log, global_log


class MetricRecorder:
    def __init__(self, phase, sparse_num=4):
        self.metrics_name = ["Top-1", "Top-5", "loss"]
        self.phase = phase
        self.recorder = [AverageMeter() for _ in self.metrics_name]
        self.spatial_sparsity_records = [AverageMeter() for _ in range(sparse_num)]

    def init(self, file_path):

        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write("Epoch\t" + "\t".join(self.metrics_name) + "\n")

    def update(self, metrics, list_metrics, inp):
        for metric, recorder in zip(metrics, self.recorder):
            recorder.update(metric.item(), inp.size(0))

        # [s_percents] = list_metrics
        # for s_per, recorder in zip(s_percents, self.spatial_sparsity_records):
        #     recorder.update(s_per, 1)

    def get_metric(self, name, current=True):
        if current:
            return self.recorder[self.metrics_name.index(name)].avg

    def get_record(self, name, current=True):
        if current:
            return self.recorder[self.metrics_name.index(name)]

    def summary(self, epoch, msg_file="", tb="", print_info=True):

        self.metrics = [recorder.avg for recorder in self.recorder]
        # spatial_layer_str = ", ".join([str(round(recorder.avg, 2)) for recorder in self.spatial_sparsity_records])
        if print_info:
            print(f'* Epoch {epoch} - Prec@1 {self.get_metric("Top-1"):.3f} - Prec@5 {self.get_metric("Top-5"):.3f} '
                  f'- Loss {self.get_metric("loss"):.3f}')

        return self.metrics


class ErrorAnalyser:
    def __init__(self, path):
        self.file = open(path, "w")
        self.sample_results = [["sample", "target", "pred", "possi", "target_possi"]]
        self.sample_cnt = 0

    def update(self, outputs, targets, sample_names=None):
        possibs, preds, target_pos = self.extract_meta(outputs, targets)
        if sample_names is None:
            sample_names = [idx+self.sample_cnt for idx in range(len(outputs))]
        else:
            sample_names = list(map(lambda x: x.split("/")[-1], sample_names))
        for sample_name, possib, pred, target, t_pos in \
                zip(sample_names, possibs, preds, targets, target_pos):
            self.sample_results.append(list(map(lambda x:str(x.cpu().tolist()) if isinstance(x, torch.Tensor) else str(x),
                                                [sample_name, target, pred, possib, t_pos])))
            self.sample_cnt += 1

    def extract_meta(self, outputs, targets):
        pos = torch.softmax(outputs, dim=1)
        preds = torch.max(outputs, dim=1)[1]
        target_pos = []
        for p, target in zip(pos, targets):
            target_pos.append(p[target])
        return torch.max(pos, dim=1)[0], preds, torch.Tensor(target_pos).cuda()

    def finish(self):
        for sample_result in self.sample_results:
            self.file.write(" ".join(sample_result))
            self.file.write("\n")
        self.file.close()
