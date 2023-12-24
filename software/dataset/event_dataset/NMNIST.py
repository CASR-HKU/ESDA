import h5py
from .base import BaseDataset
import numpy as np
import os
from .utils import read_mnist_file


class NMNISTDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, dataset_percentage=1, object_classes="all", **kwargs):
        super().__init__()
        self.name = "NMNIST"
        self.mode_name = self.get_mode_name(mode)
        self.data_percentage = dataset_percentage
        self.root = os.path.join(root, self.mode_name)

        self.object_classes = [
            "0 - zero",
            "1 - one",
            "2 - two",
            "3 - three",
            "4 - four",
            "5 - five",
            "6 - six",
            "7 - seven",
            "8 - eight",
            "9 - nine",
        ]

        self.load()
        self.common_preprocess(shuffle)

    def get_mode_name(self, mode):
        return "Train" if mode == 'training' else "Test"

    def load(self):
        for path, dirs, files in os.walk(self.root):
            files.sort()
            for idx, file in enumerate(files):
                if (idx * self.data_percentage) % 1 != 0:
                    continue
                if file.endswith("bin"):
                    self.files.append(path + "/" + file)
                    label_number = int(path[-1])
                    self.labels.append(label_number)

    def __getitem__(self, idx):
        label = self.labels[idx]
        orig_events = read_mnist_file(self.files[idx])
        events = np.zeros((orig_events.shape[0], 4), dtype=np.float32)
        for i, (x, y, t, p) in enumerate(orig_events):
            events[i, 2] = t
            events[i, 0] = x
            events[i, 1] = y
            events[i, 3] = 1 if p else 0
        # events = np.load(filename).astype(np.float32)
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label


class NMNISTProcessedDataset(NMNISTDataset):
    def __init__(self, root, mode='training', shuffle=True, dataset_percentage=1, min_event=0, **kwargs):
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
