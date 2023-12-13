import numpy as np


class Shifter:
    def __init__(self, width, height, max_shift=20, max_trial=3, **kwargs):
        self.width, self.height = width, height
        self.max_shift = max_shift
        self.max_trial = max_trial

    def __call__(self, events, bounding_box=None):
        trial = 0
        original_event = events
        while True:
            x_shift, y_shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=(2,))

            events[:, 0] += x_shift
            events[:, 1] += y_shift

            valid_events = (events[:, 0] >= 0) & (events[:, 0] < self.width) &\
                           (events[:, 1] >= 0) & (events[:, 1] < self.height)
            events = events[valid_events]
            if trial > self.max_trial:
                return original_event

            trial += 1
            if len(events) > 0:
                return events

