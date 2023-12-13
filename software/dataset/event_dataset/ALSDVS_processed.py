from os import listdir
from .base import BaseDataset
import numpy as np
import os


class ASLDVSProcessedDataset(BaseDataset):
    def __init__(self, root, mode='training', shuffle=True, cross_valid=4, **kwargs):
        super().__init__()
        self.mode_name = "train" if mode == 'training' else "test"
        self.root = root
        self.object_classes = [chr(letter) for letter in range(97, 123)]
        self.int_classes = dict(zip(self.object_classes, range(len(self.object_classes))))
        self.load()
        self.common_preprocess(shuffle)
        self.cross_valid(cross_valid)

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
            file_idx = int(file.split("/")[-1].split("_")[-1].split(".")[0])
            if self.mode_name == "train":
                if file_idx < lower or file_idx >= upper:
                    files.append(file)
                    labels.append(label)
            else:
                if lower <= file_idx < upper:
                    files.append(file)
                    labels.append(label)
        self.files, self.labels = files, labels


