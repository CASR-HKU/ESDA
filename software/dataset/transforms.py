import numpy as np
import yaml
import torch
try:
    from .transform_func import *
except:
    from transform_func import *

transform_mapping = {
    "denoise": Denoiser,
    "sample": Sampler,
    "shift": Shifter,
    "flip": Flipper,
}


class Transform:
    def __init__(self, cfg, isTrain, **kwargs):
        self.valid_transform = [k for idx, k in enumerate(cfg) if idx > 2][:-1]
        if not isTrain:
            if "shift" in self.valid_transform:
                self.valid_transform.pop(self.valid_transform.index("shift"))
            if "flip" in self.valid_transform:
                self.valid_transform.pop(self.valid_transform.index("flip"))

        height, width, raw_size = cfg["height"], cfg["width"], cfg["raw_size"]
        self.transform_name = None
        self.transforms = [transform_mapping[func](height=height, width=width, istrain=isTrain, **cfg[func])
                           for func in self.valid_transform]
        self.to_hist = HistogramGenerator(height, width, **cfg["toImage"])
        self.interplote_size = np.array([height, width]) if raw_size else self.round_input_size([height, width])

    def round_input_size(self, input_size):
        input_size = np.array(input_size) // 32
        for i in range(5):
            input_size = 2 * input_size + 1
        return input_size

    def get_sampler(self):
        if "sample" in self.valid_transform:
            return self.transforms[self.valid_transform.index("sample")]
        else:
            return None

    def process(self, events):
        for transform in self.transforms:
            events = transform(events)
        histogram = self.to_hist(events)
        # I guess it's very important to add this here
        histogram = torch.from_numpy(histogram)
        histogram = self.to_hist.hist_denoise(histogram)
        histogram = torch.nn.functional.interpolate(histogram.permute(2, 0, 1).unsqueeze(0), size=torch.Size(self.interplote_size))
        histogram = histogram.squeeze(0).permute(1, 2, 0)
        # histogram = torch.nn.functional.interpolate(histogram.permute(2, 0, 1). , size=torch.Size(self.interplote_size))
        # histogram = histogram.permute(1, 2, 0)
        return histogram, events


if __name__ == '__main__':
    setting_file = "config/new_training_settings/Ncal_settings_mobilenet.yaml"
    with open(setting_file, 'r') as stream:
        settings = yaml.load(stream, yaml.Loader)

    t = Transform(settings["transform"])
    np_path = "/home/baoheng/Documents/tonic/data/DVSGesture/ibmGestureTest/user24_fluorescent/1.npy"
    events = np.load(np_path).astype(np.float32)
    jntm = t.process(events)
    a = 1