import torch
import numpy as np

from torch.utils.data.dataloader import default_collate


class Loader:
    def __init__(self, dataset, batch_size, num_workers, pin_memory, device, shuffle=True):
        self.device = device
        split_indices = list(range(len(dataset)))
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(split_indices)
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=minkowski_collate_fn)
        else:
            self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=pin_memory,
                                                      collate_fn=minkowski_collate_fn)
    def __iter__(self):
        for data in self.loader:
            # data = [d.to(self.device) for d in data]
            yield data

    def __len__(self):
        return len(self.loader)


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


def concat_hist(data):
    hists = []
    for d in data:
        if len(hists) == 0:
            hists = d.permute(2, 0, 1).unsqueeze(0)
        else:
            hists = torch.cat([hists, d.permute(2, 0, 1).unsqueeze(0)], 0)
    return hists


def minkowski_collate_fn(list_data):
    import MinkowskiEngine as ME
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        # [d["histogram"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
        "histograms": concat_hist([d["histogram"] for d in list_data])
    }