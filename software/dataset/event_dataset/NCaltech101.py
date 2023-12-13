from os import listdir
from os.path import join
from  .base import BaseDataset
import numpy as np
import os
import h5py


class NCaltech101Dataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, dataset_percentage=1, object_classes="all", **kwargs):
        super().__init__()
        self.mode_name = self.get_mode_name(mode)
        self.root = os.path.join(root, self.mode_name)
        self.data_percentage = dataset_percentage

        self.object_classes = self.get_object_classes(object_classes)
        self.object_classes.sort()

        self.load()
        self.common_preprocess(shuffle)

    def get_object_classes(self, object_classes):
        return listdir(self.root) if object_classes == 'all' else object_classes

    def load(self):
        for i, object_class in enumerate(self.object_classes):
            if (i * self.data_percentage) % 1 != 0:
                continue
            new_files = [join(self.root, object_class, f) for f in listdir(join(self.root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        filename = self.files[idx]
        events = np.load(filename).astype(np.float32)
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label

    def get_mode_name(self, mode):
        return "training" if mode == 'training' else "validation"


class NCaltech101ProcessedDataset(NCaltech101Dataset):
    def __init__(self, root, mode='training', shuffle=True, dataset_percentage=1, min_event=1, **kwargs):
        self.data_percentage = dataset_percentage
        self.min_event = min_event
        super().__init__(root, mode=mode, shuffle=shuffle, **kwargs)

    def load(self):
        with h5py.File(self.root, "r") as f:
            for idx, name in enumerate(f.keys()):
                if (idx * self.data_percentage) % 1 != 0:
                    continue
                data = f[name]["data"][:]
                if len(data) < self.min_event:
                    continue
                self.files.append(data)
                self.labels.append(f[name]["label"][()])
            # self.labels.append( np.load(os.path.join(self.root, file), allow_pickle=True).item()["label"])

    def __getitem__(self, idx):
        label = self.labels[idx]
        # orig_events = read_mnist_file(self.files[idx])
        events = self.files[idx]
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label

    def get_mode_name(self, mode):
        return "train.h5" if mode == 'training' else "val.h5"

    def get_object_classes(self, object_classes):
        label_file = os.path.join('/'.join(self.root.split("/")[:-1]), "labels.txt")
        with open(label_file, 'r') as f:
            lines = [line[:-1] for line in f.readlines()]
        return lines