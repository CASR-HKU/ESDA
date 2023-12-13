

from os import listdir
from .base import BaseDataset
import numpy as np
import os


class PokerDVSDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, object_classes="all", **kwargs):
        super().__init__()
        self.mode_name = "pips_train" if mode == 'training' else "pips_test"
        self.root = os.path.join(root, self.mode_name)

        self.classes = ["cl", "he", "di", "sp"]
        self.int_classes = dict(zip(self.classes, range(4)))
        # self.object_classes.sort()

        self.load()
        self.common_preprocess(shuffle)

    def load(self):
        for path, dirs, files in os.walk(self.root):
            files.sort()
            for file in files:
                if file.endswith("npy"):
                    self.files.append(np.load(os.path.join(path, file)))
                    self.labels.append(self.int_classes[path[-2:]])

    def __getitem__(self, idx):
        raw_events, label = self.files[idx], self.labels[idx]
        # events = np.load(filename).astype(np.float32)
        events = np.zeros((raw_events.shape[0], 4), dtype=np.float32)
        for i, (t,x,y,p) in enumerate(raw_events):
            events[i,2] = t
            events[i,0] = x
            events[i,1] = y
            events[i,3] = p
        events[:, 3][np.where(events[:, 3] == -1)[0]] = 0
        return events, label


