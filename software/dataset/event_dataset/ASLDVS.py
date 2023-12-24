from collections import defaultdict
import numpy as np
from .base import BaseDataset
import os
import scipy.io as scio
import h5py


class ASLDVSDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, cross_valid=4, **kwargs):
        super().__init__()
        self.mode_name = self.get_mode_name(mode)
        self.root = root
        self.name = "ASL"
        self.object_classes = [chr(letter) for letter in range(97, 123)]
        self.int_classes = dict(zip(self.object_classes, range(len(self.object_classes))))
        self.load()
        self.common_preprocess(shuffle)
        self.cross_valid(cross_valid)

    def load(self):
        self.class_sample = defaultdict(int)
        for path, dirs, files in os.walk(self.root):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    self.class_sample[file.split("_")[0]] += 1
                    self.files.append(path + "/" + file)
                    self.labels.append(self.int_classes[path[-1]])

    def __getitem__(self, idx):
        events, label = scio.loadmat(self.files[idx]), self.labels[idx]
        events = np.concatenate((events["x"], events["y"], events["ts"], events["pol"]), axis=1)
        # label = self.labels[idx]
        return events.astype(np.float32), label

    def get_file_idx(self, file):
        return int(file.split("/")[-1].split("_")[-1].split(".")[0])

    def cross_valid(self, idx):
        if idx == -1:
            print("Not using cross validation. Train and test set are the same")
            return
        print("Cross validation: The sample {} are validation set".format(idx))
        sample_ratio = 0.2
        cls_size = 4200
        bound = sample_ratio * idx
        lower_bound = int(bound * cls_size)
        upper_bound = int((bound + sample_ratio) * cls_size)
        self.select_sample(lower_bound, upper_bound)

    def select_sample(self, lower, upper):
        files, labels = [], []
        for file, label in zip(self.files, self.labels):
            file_idx = self.get_file_idx(file)
            if self.mode_name == "train":
                if file_idx < lower or file_idx >= upper:
                    files.append(file)
                    labels.append(label)
            else:
                if lower <= file_idx < upper:
                    files.append(file)
                    labels.append(label)
        self.files, self.labels = files, labels

    def get_mode_name(self, mode):
        return "train" if mode == 'training' else "validation"


class ASLDVSProcessedDataset(ASLDVSDataset):
    def __init__(self, root, mode='training', shuffle=True, cross_valid=4, dataset_percentage=1, min_event=0, **kwargs):
        self.data_percentage = dataset_percentage
        self.min_event = min_event
        super().__init__(root, mode=mode, shuffle=shuffle, cross_valid=cross_valid, **kwargs)

    def load(self):
        self.names = []
        with h5py.File(os.path.join(self.root, "data.h5"), "r") as f:
            for idx, name in enumerate(f.keys()):
                if (idx * self.data_percentage) % 1 != 0:
                    continue
                data = f[name]["data"][:]
                if len(data) < self.min_event:
                    continue
                if data[:,0].max() > 239 or data[:,1].max() > 179:
                    continue
                self.files.append(data)
                self.names.append(name)
                self.labels.append(f[name]["label"][()])

    def __getitem__(self, idx):
        label = self.labels[idx]
        events = self.files[idx]
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label

    def get_file_idx(self, file):
        return int(file.split("/")[-1].split("-")[0].split("_")[1])

    def select_sample(self, lower, upper):
        files, labels = [], []
        for file, label, name in zip(self.files, self.labels, self.names):
            file_idx = self.get_file_idx(name)
            if self.mode_name == "train":
                if file_idx < lower or file_idx >= upper:
                    files.append(file)
                    labels.append(label)
            else:
                if lower <= file_idx < upper:
                    files.append(file)
                    labels.append(label)
        self.files, self.labels = files, labels
