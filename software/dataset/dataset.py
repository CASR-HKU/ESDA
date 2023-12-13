try:
    from .transforms import Transform
except:
    from transforms import Transform
import numpy as np
import torch
from copy import deepcopy
try:
    import dataset.event_dataset as ds
except:
    import event_dataset as ds


def round_input_size(input_size):
    input_size = np.array(input_size) // 32
    for i in range(5):
        input_size = 2 * input_size + 1
    return input_size


class Dataset:
    def __init__(self, cfg, mode='training', shuffle=False, **kwargs):
        self.mode = mode
        self.transform = Transform(cfg["transform"], isTrain=True, **kwargs) if mode == "training" \
            else Transform(cfg["transform"], isTrain=False, **kwargs)

        self.dataset_name = cfg["dataset"]["name"]
        self.dataset_type = "event"
        if self.dataset_name == "NCaltech101":
            self.dataset = ds.NCal(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "DVSGesture_processed":
            self.dataset = ds.DVS(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "DVSPreprocessed":
            self.dataset = ds.DVSProcessed(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "DVSLip":
            self.dataset = ds.Lip(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "PokerDVS":
            self.dataset = ds.Poker(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "ASLDVS":
            self.dataset = ds.ASL(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "NMNIST":
            self.dataset = ds.NMNIST(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "NMNISTProcessed":
            self.dataset = ds.NMNISTProcessed(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "ASLDVSProcessed":
            self.dataset = ds.ASLProcessed(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "NCaltech101Processed":
            self.dataset = ds.NCalProcessed(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
        elif self.dataset_name == "CIFAR10":
            self.dataset = ds.CIFAR10(cfg["dataset"]["dataset_path"], mode, shuffle,
                                      sampler=self.transform.get_sampler(), **kwargs)
        elif self.dataset_name == "IniRoshambo":
            self.dataset = ds.IniRoshambo(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
            self.dataset_type = "frame"
        elif self.dataset_name == "DVS_slice":
            self.dataset = ds.DVSSlice(cfg["dataset"]["dataset_path"], mode, shuffle, **kwargs)
            self.dataset_type = "frame"
        else:
            raise NotImplementedError
        self.nr_classes = self.dataset.nr_classes

    def __len__(self):
        return len(self.dataset.files)

    def dense_to_sparse(self, dense):
        non_zero_indices = torch.nonzero(torch.abs(dense).sum(axis=-1))
        select_indices = non_zero_indices.split(1, dim=1)
        features = torch.squeeze(dense[select_indices], dim=-2)
        return  non_zero_indices, features

    def load_sample(self, idx):
        return self.dataset[idx]

    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        if self.dataset_type == "event":
            raw_events, label = self.dataset[idx]
            histogram, event = self.transform.process(deepcopy(raw_events))
        elif self.dataset_type == "frame":
            raw_events, event = None, None
            histogram, label = self.dataset[idx]
            histogram = torch.from_numpy(histogram.copy()).float().permute(1, 2, 0)
            # histogram, event = self.transform.process(deepcopy(raw_events))
        else:
            raise NotImplementedError
        coords, features = self.dense_to_sparse(histogram)

        return {
            "raw_event": raw_events,
            "event": event,
            "coordinates": coords,
            "features": features,
            "label": label,
            "histogram": histogram
        }
        
        # return events, label, histogram

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
    import argparse

    parser = argparse.ArgumentParser(description='Train network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)
    parser.add_argument('--data_path', help='Path to dataset', default="")
    # parser.add_argument('--val', action='store_true', help='Load validation set; Default training')
    parser.add_argument('--min_event', type=int, default=0)
    args = parser.parse_args()

    with open(args.settings_file, 'r') as stream:
        settings = yaml.load(stream, yaml.Loader)

    if args.data_path != "":
        settings["dataset"]["dataset_path"] = args.data_path

    dataset_name = settings["dataset"]["name"]
    modes = ["validation"] if "ASLDVS" not in dataset_name else ["training"]

    for mode in modes:
        d = Dataset(settings, mode=mode, min_event=args.min_event, cross_valid=-1)
        # times, events_num = [], []
        # for idx in range(len(d)):
        #     event = d.load_sample(idx)[0]
        #     events_num.append(event.shape[0])
        #     times.append(event[-1][2] - event[0][2])
        #
        # def normalize(array):
        #     array_min, array_max = np.min(array), np.max(array)
        #     return (array - array_min) / (array_max - array_min)
        #
        # events_num = np.array(events_num)
        # times = np.array(times)
        # print("Mode: {}".format(mode))
        # print("Time maximum: {}".format(np.max(times)))
        # print("Time minimum: {}".format(np.min(times)))
        # print("Time mean: {}".format(np.mean(times)))
        #
        # print("Events maximum: {}".format(np.round(np.max(events_num)), 4))
        # print("Events minimum: {}".format(np.round(np.min(events_num)), 4))
        # print("Events mean: {}".format(np.round(np.mean(events_num)), 4))



