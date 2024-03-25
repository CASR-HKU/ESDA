import random
import numpy as np
from collections import Counter



class SpatialMask:
    def __init__(self, p=0.5, ratio=0.1, **kwargs):
        self.p = p
        self.ratio = ratio

    def __call__(self, images, label):
        if np.random.rand() < self.p:
            nonzeros = np.nonzero(np.sum(images, axis=1))
            nonzero_per_img = self.count_numbers(nonzeros[0])
            selected_pixels = [random.sample(list(range(i)), int(i * self.ratio)) for i in nonzero_per_img]
            begin_idx = 0
            for img_idx in range(len(images)):
                pixels = selected_pixels[img_idx]
                for pixel_idx, pixel in enumerate(pixels):
                    h_idx, w_idx = nonzeros[1][pixel_idx+begin_idx], nonzeros[2][pixel_idx+begin_idx]
                    images[img_idx, :, h_idx, w_idx] = 0
                begin_idx += nonzero_per_img[img_idx]
        return images, label

    def count_numbers(self, lst):
        counter = Counter(lst)
        counts = [counter[i] for i in range(lst[-1]+1)]
        return counts


class TemporalMask:
    def __init__(self, p=0.5, max_length=5, **kwargs):
        self.p = p
        self.max_length = max_length

    def __call__(self, images, label):
        if np.random.rand() > self.p:
            self.masked_temporal_length = min(abs(int(random.gauss(0, 0.5) * self.max_length)), self.max_length)
            if self.masked_temporal_length > 0:
                images[:self.masked_temporal_length, :, :, :] = 0
                label[: self.masked_temporal_length, :] = 0
        return images, label


class DropEvent:
    def __init__(self, **kwargs):
        pass

    def __call__(self, images, label):
        return images, label


class Flip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, label):
        if random.random() > 1 - self.p:
            images = np.flip(images, axis=-1)
            label[:, 0] = 1 - label[:, 0]
        return np.ascontiguousarray(images), label


class Shift:
    def __init__(self, p=0.5, max_shift=10):
        self.p = p
        self.max_shift = max_shift
        self.max_choose = 3

    def shift_array(self, arr, x, y):
        t, c, h, w = arr.shape
        padded_arr = np.pad(arr, ((0, 0), (0, 0), (self.max_shift, self.max_shift), (self.max_shift, self.max_shift)), mode='constant')
        padded_arr = np.roll(padded_arr, (x, y), axis=(2, 3))
        return padded_arr[:, :, self.max_shift:h + self.max_shift, self.max_shift:w + self.max_shift]

    def __call__(self, images, label):
        _, _, self.height, self.width = images.shape
        if random.random() > 1 - self.p:
            shift_choose = 0
            while True:
                if shift_choose > self.max_choose:
                    shift_x, shift_y = 0, 0
                    break
                else:
                    shift_x, shift_y = (random.randint(-self.max_shift, self.max_shift),
                                        random.randint(-self.max_shift, self.max_shift))
                    if (self.check_in_range(label[:, 0] + shift_x/self.width) and
                            self.check_in_range(label[:, 1] + shift_y/self.height)):
                        break
                    shift_choose += 1

            label[:, 0] += shift_x/self.width
            label[:, 1] += shift_y/self.height

            images = self.shift_array(images, shift_x, shift_y)
        return images, label


    @staticmethod
    def check_in_range(p, pmax=1, pmin=0):
        higher_bound = p <= pmax
        lower_bound = p >= pmin
        if not higher_bound.all() or not lower_bound.all():
            return False
        return True
