from os import listdir
import pickle
from .base import BaseDataset
import os
import h5py
import numpy as np


class DVSGestureDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=False, dataset_percentage=1, **kwargs):
        super().__init__()
        self.name = "DVS"
        self.data_percentage = dataset_percentage
        self.object_classes = ["hand clapping", "right hand wave", "left hand wave", "right arm clockwise",
                               "right arm counter clockwise", "left arm clockwise", "left arm counter clockwise",
                               "arm roll", "air drums", "air guitar"]
        self.mode_name = self.get_mode_name(mode)
        self.data = []
        self.root = os.path.join(root, self.mode_name)
        self.load()
        self.common_preprocess(shuffle)

    def get_mode_name(self, mode):
        return "train" if mode == 'training' else "test"

    def load(self):
        files = listdir(self.root)
        for idx, file in enumerate(files):
            if (idx * self.data_percentage) % 1 != 0:
                continue
            # self.files.append(file)
            with open(os.path.join(self.root, file), 'rb') as f:
                dataset = pickle.load(f)
            self.files += [file.replace(".pkl", "_{}.pkl".format(idx)) for idx in range(len(dataset['data']))]
            self.data += dataset['data']
            self.labels += dataset['label'].tolist()

    def __getitem__(self, idx):
        label = self.labels[idx]
        events = self.data[idx]
        return events, label


class DVSGestureProcessedDataset(DVSGestureDataset):
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
