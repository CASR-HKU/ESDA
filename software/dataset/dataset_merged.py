import os
import random
from os import listdir
from os.path import join
try:
    from .transforms import Transform
except:
    from transforms import Transform
import numpy as np
import pickle
import torch


def round_input_size(input_size):
    input_size = np.array(input_size) // 32
    for i in range(5):
        input_size = 2 * input_size + 1
    return input_size


class Dataset:
    def __init__(self, cfg, mode='training', shuffle=True, **kwargs):
        self.mode = mode
        self.transform = Transform(cfg["transform"], isTrain=True, **kwargs) if mode == "training" \
            else Transform(cfg["transform"], isTrain=False, **kwargs)

        self.dataset_name = cfg["dataset"]["name"]
        if self.dataset_name == "NCaltech101":
            self.mode_name = "training" if self.mode == 'training' else "validation"
        elif self.dataset_name == "DVSGesture":
            self.mode_name = "ibmGestureTrain" if self.mode == 'training' else "ibmGestureTest"
        elif self.dataset_name == "DVSGesture_processed":
            self.mode_name = "train" if self.mode == 'training' else "test"
        else:
            raise NotImplementedError

        object_classes = cfg["dataset"]["object_classes"]
        root = os.path.join(cfg["dataset"]["dataset_path"], self.mode_name)
        print("dataset dir", root)

        self.object_classes = listdir(root) if object_classes == 'all' else object_classes
        if "DVSGesture" in self.dataset_name:
            self.object_classes = ["hand clapping", "right hand wave", "left hand wave", "right arm clockwise", "right arm counter clockwise", "left arm clockwise", "left arm counter clockwise", "arm roll", "air drums", "air guitar", "other gestures",]
        self.object_classes.sort()
        # self.nr_classes = len(self.object_classes) if "DVSGesture" not in self.dataset_name else 11

        self.files = []
        self.labels = []
        if self.dataset_name == "NCaltech101":
            self.load_Ncal(root)
        elif self.dataset_name == "DVSGesture":
            self.load_DVS(root)
        elif self.dataset_name == "DVSGesture_processed":
            self.load_DVS_processed(root)
        else:
            raise NotImplementedError
        self.nr_samples = len(self.labels)
        print("Total sample num is {}".format(self.nr_samples))
        self.nr_classes = max(self.labels) + 1

        if shuffle:
            zipped_lists = list(zip(self.files, self.labels))
            random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.labels = zip(*zipped_lists)

    def __len__(self):
        return len(self.files)

    def dense_to_sparse(self, dense):
        non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense[select_indices], dim=-2)
        return  non_zero_indices, features   

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """

        # if "processed" not in self.dataset_name:
        label = self.labels[idx]
        if "processed" not in self.dataset_name:
            filename = self.files[idx]
            events = np.load(filename).astype(np.float32)
        else:
            events = self.files[idx]
        if self.dataset_name == "DVSGesture":
            events = events[:, [0, 1, 3, 2]]
        elif self.dataset_name == "NCaltech101":
            events[:, 3][np.where(events[:, 3] == -1)[0]] = 0

        histogram = self.transform.process(events)
        
        # histogram = torch.from_numpy(histogram)
        coords, features = self.dense_to_sparse(histogram)

        return {
            "coordinates": coords,
            "features": features,
            "label": label
        }
        
        # return events, label, histogram

    def load_DVS_processed(self, root):
        files = listdir(root)
        self.data, self.labels = [], []
        for f in files:
            with open(os.path.join(root, f), 'rb') as f:
                dataset = pickle.load(f)
            self.files += dataset['data']
            self.labels += dataset['label'].tolist()

    def load_Ncal(self, root):
        for i, object_class in enumerate(self.object_classes):
            new_files = [join(root, object_class, f) for f in listdir(join(root, object_class))]
            self.files += new_files
            self.labels += [i] * len(new_files)

    def load_DVS(self, root):
        file_path = os.path.join(root)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("npy"):
                    self.files.append(path + "/" + file)
                    self.labels.append(int(file[:-4]))

    def preprocess(self, window, overlap_ratio, preprocess_type="time", **kwargs):
        if preprocess_type == "time":
            from .preprocess import preprocess_time as preprocess
        elif preprocess_type == "event":
            from .preprocess import preprocess_event as preprocess
        else:
            raise NotImplementedError
        preprocess(self, window, overlap_ratio, **kwargs)


if __name__ == '__main__':
    import yaml
    setting_file = "config/new_training_settings/DVS_settings_mobilenet.yaml"
    with open(setting_file, 'r') as stream:
        settings = yaml.load(stream, yaml.Loader)

    d = Dataset(settings, mode="validation")
    times, events_num = [], []
    for idx in range(len(d)):
        events, jini, taimei = d[idx]
        events_num.append(events.shape[0])
        times.append(events[-1][2])

    def normalize(array):
        array_min, array_max = np.min(array), np.max(array)
        return (array - array_min) / (array_max - array_min)

    events_num = np.array(events_num)
    times = np.array(times)
    print("Time maximum: {}".format(np.max(times)))
    print("Time normalized variance: {}".format(np.var(normalize(times))))
    print("Time mean: {}".format(np.mean(times)))

    print("Events maximum: {}".format(np.round(np.max(events_num)), 4))
    print("Events normalized variance: {}".format(np.var(normalize(events_num))))
    print("Events mean: {}".format(np.round(np.mean(events_num)), 4))



