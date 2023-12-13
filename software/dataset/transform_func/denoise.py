import numpy as np


class Denoiser:
    def __init__(self, height, width, filter_time=10000, **kwargs):
        self.filter_time = filter_time
        self.width, self.height = width, height

    def __call__(self, events, filter_time=-1):
        if self.filter_time == -1:
            return events
        if filter_time == -1:
            filter_time = self.filter_time
        events_copy = np.zeros_like(events)
        copy_index = 0
        timestamp_memory = np.zeros((self.width, self.height)) + filter_time

        for event in events:
            x, y, t, _ = event
            x, y = int(x), int(y)
            timestamp_memory[x, y] = t + filter_time
            if (
                    (x > 0 and timestamp_memory[x - 1, y] > t)
                    or (x < self.width - 1 and timestamp_memory[x + 1, y] > t)
                    or (y > 0 and timestamp_memory[x, y - 1] > t)
                    or (y < self.height - 1 and timestamp_memory[x, y + 1] > t)
            ):
                events_copy[copy_index] = event
                copy_index += 1

        return events_copy[:copy_index]