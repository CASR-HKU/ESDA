from os import listdir
from os.path import join
from .base import BaseDataset
import numpy as np
import os
from .utils import read_mnist_file


class NMNISTProcessedDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, object_classes="all", **kwargs):
        super().__init__()
        self.mode_name = "train" if mode == 'training' else "test"
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

    def load(self):
        files = listdir(self.root)
        for file in files:
            if file == "labels.txt":
                continue
            content = np.load(os.path.join(self.root, file), allow_pickle=True).item()
            self.files.append(content['event'])
            self.labels.append(content['label'])

    def __getitem__(self, idx):
        label = self.labels[idx]
        # orig_events = read_mnist_file(self.files[idx])
        events = self.files[idx] #np.load(self.files[idx], allow_pickle=True).item()
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label


