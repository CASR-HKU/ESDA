import random
import numpy as np


class Sampler:
    def __init__(self, window=10000, float=0, istrain=True, sample_type="continuous", **kwargs):
        self.sample_window = window
        self.sample_float = float
        self.isTrain = istrain
        self.sample_type = sample_type
        assert self.sample_type in ["continuous", "random"], "sample_type can not be {}".format(self.sample_type)

    def shuffle_events(self, events):
        idx = np.arange(events.shape[0])
        np.random.shuffle(idx)
        # idx_full = np.zeros(self.sample_window, dtype=int)
        return events[idx, ...]

    def __call__(self, events):
        nr_events = events.shape[0]
        if nr_events < self.sample_window:
            return events
        window_size = int(self.sample_window * (1+random.uniform(-self.sample_float, self.sample_float))) \
            if self.isTrain else self.sample_window

        if self.sample_type == "random":
            events = self.shuffle_events(events)

        while True:
            window_start = 0 if not self.isTrain else random.randrange(0, max(1, nr_events - window_size))
            # window_size = self.sample_window * random.randrange(-self.sample_float, self.sample_float)
            window_end = min(nr_events, window_start + window_size)
            try:
                sampled_events = events[window_start:window_end, :]
            except:
                sampled_events = events[window_start:window_end]
            if len(sampled_events):
                return sampled_events
