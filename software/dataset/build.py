import torch
import numpy as np
from torch.utils.data.dataloader import default_collate


def collate_events(data):
    labels = []
    events = []
    histograms = []
    for i, d in enumerate(data):
        labels.append(d[1])
        histograms.append(d[2])
        ev = np.concatenate([d[0], i*np.ones((len(d[0]), 1), dtype=np.float32)], 1)
        events.append(ev)
    events = torch.from_numpy(np.concatenate(events, 0))
    labels = default_collate(labels)

    histograms = default_collate(histograms)

    return events, labels, histograms


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device, shuffle=True):
        self.device = device
        split_indices = list(range(len(dataset)))
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=collate_events)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=collate_events)
    def __iter__(self):
        for data in self.loader:
            data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


def build_dataset(cfg_file):
    from dataset.dataset import Dataset
    import yaml
    with open(cfg_file, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)
    train_dataset = Dataset(cfg, mode="training")

    # nr_train_epochs = int(train_dataset.nr_samples / self.settings.batch_size) + 1
    nr_classes = train_dataset.nr_classes
    object_classes = train_dataset.object_classes

    val_dataset = Dataset(cfg, mode="validation")

    # test_dataset = self.dataset_builder(self.settings.dataset_path,
    #                                    self.settings.object_classes,
    #                                    self.settings.height,
    #                                    self.settings.width,
    #                                    self.settings.nr_events_window,
    #                                    mode='testing',
    #                                    event_representation=self.settings.event_representation)

    # nr_val_epochs = int(val_dataset.nr_samples / self.settings.batch_size) + 1

    train_loader = dataset_loader(train_dataset, batch_size=self.settings.batch_size,
                                            device=self.settings.gpu_device,
                                            num_workers=self.settings.num_cpu_workers, pin_memory=False)
    val_loader = dataset_loader(val_dataset, batch_size=self.settings.batch_size,
                                          device=self.settings.gpu_device,
                                          num_workers=self.settings.num_cpu_workers, pin_memory=False)
