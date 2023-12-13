import numpy as np


class Flipper:
    def __init__(self, height, width, p=0, **kwargs):
        self.width, self.height = width, height
        self.flip_p = p

    def __call__(self, events, bounding_box=None):
        flipped = False
        if np.random.random() < self.flip_p:
            events[:, 0] = self.width - 1 - events[:, 0]
            flipped = True

        if bounding_box is None:
            return events

        if flipped:
            bounding_box[:, 0] = self.width - 1 - bounding_box[:, 0]
            bounding_box = bounding_box[[1, 0, 3, 2]]
        return events, bounding_box
