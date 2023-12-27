from .base import BaseDataset
import numpy as np
import os
import lmdb
from .datum_pb2 import Datum


class Transform:
    def __init__(self, mode="training", flip_ratio=0.5, max_shift=10, shift_ratio=0.5):
        self.mode = mode
        self.flip_ratio = flip_ratio
        self.max_shift = max_shift
        self.shift_ratio = shift_ratio

    def __call__(self, data):
        if self.mode == "training":
            data = self.flip(data) if np.random.rand() < self.flip_ratio else data
            shift = np.random.randint(-self.max_shift, self.max_shift)
            data = self.shift(data, shift) if np.random.rand() < self.shift_ratio else data
        return data

    def flip(self, data):
        return np.flip(data, axis=1)

    def shift(self, data, shift):
        return np.roll(data, shift, axis=1)


class RoshamboDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, dataset_percentage=1, object_classes="all", **kwargs):
        super().__init__()
        self.mode_name = self.get_mode_name(mode)
        self.root = os.path.join(root, self.mode_name)
        self.data_percentage = dataset_percentage

        self.object_classes = ["background", "rock", "paper", "scissors"]

        self.load()
        self.common_preprocess(shuffle)
        self.transform = Transform(self.mode_name)
        self.name = "Roshambo"

    def load(self):
        lmdb_env = lmdb.open(self.root, lock=True)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()

        datum = Datum()

        for idx, (key, value) in enumerate(lmdb_cursor):
            if (idx * self.data_percentage) % 1 != 0:
                continue
            datum.ParseFromString(value)
            label = int(datum.label)
            self.labels.append(label)
            height = int(datum.height)
            width = int(datum.width)
            channel = int(datum.channels)
            image = np.frombuffer(datum.data, dtype=np.uint8).reshape((channel, height, width))
            self.files.append(image)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.files[idx]
        data = self.transform(data)
        return data, label

    def get_mode_name(self, mode):
        return "shuffled_train" if mode == 'training' else "shuffled_test"
