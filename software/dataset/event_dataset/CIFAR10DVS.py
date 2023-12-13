
from os.path import join
from .base import BaseDataset
import numpy as np
import os
from .utils import read_aedat4


class Cifar10DVSDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, cross_valid=9, object_classes="all", sampler=None, **kwargs):
        super().__init__()
        self.mode_name = "train" if mode == 'training' else "test"
        self.root = os.path.join(root)
        self.sampler = sampler

        self.classes = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }
        self.object_classes = [k for k, v in self.classes.items()]

        self.load()
        self.common_preprocess(shuffle)
        self.cross_valid(cross_valid)

    def load(self):
        file_path = os.path.join(self.root)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("aedat4"):
                    self.files.append(os.path.join(path, file))
                    label_number = self.classes[os.path.basename(path)]
                    self.labels.append(label_number)

    def __getitem__(self, idx):
        orig_events = read_aedat4(self.files[idx])
        if self.sampler is not None:
            orig_events = self.sampler(orig_events)
        # orig_events = orig_events[:1024]
        orig_events.dtype.names = ["t", "x", "y", "p"]
        label = self.labels[idx]
        events = np.zeros((orig_events.shape[0], 4), dtype=np.float32)
        for i, (t, x, y, p) in enumerate(orig_events):
            events[i, 2] = t
            events[i, 0] = x
            events[i, 1] = y
            events[i, 3] = 1 if p else 0
        # events = np.load(filename).astype(np.float32)
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label

    def cross_valid(self, idx):
        if idx == -1:
            print("Not using cross validation. Train and test set are the same")
            return
        print("Cross validation: The sample {} are validation set".format(idx))
        sample_ratio = 0.1
        cls_size = 1000
        bound = sample_ratio * idx
        lower_bound = int(bound * cls_size)
        upper_bound = int((bound + sample_ratio) * cls_size)
        self.select_sample(lower_bound, upper_bound)

    def select_sample(self, lower, upper):
        files, labels = [], []
        for file, label in zip(self.files, self.labels):
            file_idx = int(file.split("_")[-1].split(".")[0])
            if self.mode_name == "train":
                if file_idx < lower or file_idx >= upper:
                    files.append(file)
                    labels.append(label)
            else:
                if lower <= file_idx < upper:
                    files.append(file)
                    labels.append(label)
        self.files, self.labels = files, labels
