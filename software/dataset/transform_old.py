import numpy as np
import yaml
import random


class Transform:
    def __init__(self, cfg):
        mapping = {"denoise": self.denoise, "sample": self.sample, "shift": self.shift, "flip": self.random_flip}
        valid_transform = [k for idx, k in enumerate(cfg) if idx > 1][:-1]
        self.transforms = [mapping[func] for func in valid_transform]
        self.height, self.width = cfg["height"], cfg["width"]
        self.max_shift = cfg["shift"]["max_shift"]
        self.flip_p = cfg["flip"]["p"]
        self.sample_window = cfg["sample"]["window"]
        self.filter_time = cfg["denoise"]["filter_time"]

    def shift(self, events, bounding_box=None):
        if bounding_box is not None:
            x_shift = np.random.randint(-min(bounding_box[0, 0], self.max_shift),
                                        min(W - bounding_box[2, 0],self. max_shift), size=(1,))
            y_shift = np.random.randint(-min(bounding_box[0, 1], self.max_shift),
                                        min(H - bounding_box[2, 1], self.max_shift), size=(1,))
            bounding_box[:, 0] += x_shift
            bounding_box[:, 1] += y_shift
        else:
            x_shift, y_shift = np.random.randint(-self.max_shift, self.max_shift + 1, size=(2,))

        events[:, 0] += x_shift
        events[:, 1] += y_shift

        valid_events = (events[:, 0] >= 0) & (events[:, 0] < self.width) &\
                       (events[:, 1] >= 0) & (events[:, 1] < self.height)
        events = events[valid_events]

        if bounding_box is None:
            return events

        return events, bounding_box

    def generate_event_histogram(self, events):
        """
        Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
        coordinates u and v.
        """
        x, y, t, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        img_pos = np.zeros((self.height * self.width,), dtype="float32")
        img_neg = np.zeros((self.height * self.width,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + self.width * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + self.width * y[p == 0], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((self.height, self.width, 2))

        return histogram

    def random_flip(self, events, bounding_box=None):
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

    def sample(self, events):
        nr_events = events.shape[0]
        window_start = random.randrange(0, max(1, nr_events - self.sample_window))
        window_end = min(nr_events, window_start + self.sample_window)
        sampled_events = events[window_start:window_end, :]
        return sampled_events

    def denoise(self, events):

        # assert "x" and "y" and "t" in events.dtype.names

        events_copy = np.zeros_like(events)
        copy_index = 0
        width = int(events["x"].max()) + 1
        height = int(events["y"].max()) + 1
        timestamp_memory = np.zeros((width, height)) + self.filter_time

        for event in events:
            x = int(event["x"])
            y = int(event["y"])
            t = event["t"]
            timestamp_memory[x, y] = t + self.filter_time
            if (
                    (x > 0 and timestamp_memory[x - 1, y] > t)
                    or (x < width - 1 and timestamp_memory[x + 1, y] > t)
                    or (y > 0 and timestamp_memory[x, y - 1] > t)
                    or (y < height - 1 and timestamp_memory[x, y + 1] > t)
            ):
                events_copy[copy_index] = event
                copy_index += 1

        return events_copy[:copy_index]

    def process(self, events):
        for transform in self.transforms:
            events = transform(events)
        # events = self.denoise(events)
        # events = self.shift(events)
        # events = self.random_flip(events)
        # events = self.sample(events)
        histogram = self.generate_event_histogram(events)
        return histogram


if __name__ == '__main__':
    setting_file = "/home/baoheng/Documents/EventCameraNet/config/new_training_settings/DVS_settings_mobilenet.yaml"
    with open(setting_file, 'r') as stream:
        settings = yaml.load(stream, yaml.Loader)

    t = Transform(settings["transform"])
    np_path = "/home/baoheng/Documents/tonic/data/DVSGesture/ibmGestureTest/user24_fluorescent/1.npy"
    events = np.load(np_path).astype(np.float32)
    jntm = t.process(events)
    a = 1