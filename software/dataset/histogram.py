import numpy as np
import torch
import math
import cv2


class HistogramGenerator:
    def __init__(self, height, width, type="histogram", denoise=False, **kwargs):
        self.width, self.height = width, height
        self.type = type
        self.denoise = denoise

    def to_histogram(self, events):
        x, y, t, p = events.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        img_pos = np.zeros((self.height * self.width,), dtype="float32")
        img_neg = np.zeros((self.height * self.width,), dtype="float32")

        np.add.at(img_pos, x[p == 1] + self.width * y[p == 1], 1)
        np.add.at(img_neg, x[p == -1] + self.width * y[p == -1], 1)

        histogram = np.stack([img_neg, img_pos], -1).reshape((self.height, self.width, 2))
        return histogram

    def hist_denoise(self, hist):
        if not self.denoise:
            return hist
        orig_mask = (torch.sum(torch.abs(hist), dim=(2)) != 0).float().unsqueeze(0).unsqueeze(0)
        denoise_kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        denoised_mask = torch.nn.functional.conv2d(orig_mask, denoise_kernel, bias=None, stride=1, padding=1) != 0
        final_mask = (orig_mask * denoised_mask).squeeze().unsqueeze(2)
        return hist * final_mask.expand_as(hist) if final_mask.sum() else hist

    def to_timeSurface(self, events):
        tau = 5e3
        radius_x = 0
        radius_y = 0

        timestamp_memory = np.zeros(
            (2, self.height + radius_y * 2, self.width + radius_x * 2)
        )
        timestamp_memory -= tau * 3 + 1
        all_surfaces = np.zeros(
            (len(events), 2, self.height, self.width)
        )
        for index, event in enumerate(events):
            x = int(event[0])
            y = int(event[1])
            timestamp_memory[int(event[3]), y + radius_y, x + radius_x] = event[2]
            if radius_x > 0 and radius_y > 0:
                timestamp_context = (
                        timestamp_memory[
                        :, y: y + self.height, x: x + self.width
                        ]
                        - event["t"]
                )
            else:
                timestamp_context = timestamp_memory - event[2]

            timesurface = np.exp(timestamp_context / tau)
            all_surfaces[index, :, :, :] = timesurface
        return all_surfaces

    def to_leastTimeStamp(self, events):
        timestamp_memory_pos = np.zeros([self.height, self.width], dtype=np.float32)
        timestamp_memory_neg = np.zeros([self.height, self.width], dtype=np.float32)
        for x, y, t, p in events:
            if p == 1:
                timestamp_memory_pos[int(y)][int(x)] = t
            else:
                timestamp_memory_neg[int(y)][int(x)] = t
        timestamp_memory = np.dstack((timestamp_memory_pos, timestamp_memory_neg))
        sorted_timestamp = np.sort(np.unique(timestamp_memory.flatten()))
        if len(sorted_timestamp) > 2:
            t_min, t_max = sorted_timestamp[1], sorted_timestamp[-1]
            normalized_timestamp = ((timestamp_memory - t_min) * (timestamp_memory != 0).astype(int)) / (t_max - t_min)
        elif len(sorted_timestamp) == 2:
            normalized_timestamp = (timestamp_memory != 0).astype(int)
        else:
            normalized_timestamp = timestamp_memory
        return normalized_timestamp.astype(np.float32)

    def __call__(self, events):
        if self.type == "histogram":
            return self.to_histogram(events)
        elif self.type == "event_queue":
            return self.to_eventQueue(events)
        elif self.type == "time_surface":
            return self.to_timeSurface(events)
        elif self.type == "least_timestamp":
            return self.to_leastTimeStamp(events)
        else:
            raise NotImplementedError


def histogram_visualize(histogram, model_input_size, labels, path_name=None):
    histogram = torch.nn.functional.interpolate(histogram.permute(0, 3, 1, 2), torch.Size(model_input_size))
    histogram = histogram.permute(0, 2, 3, 1)

    locations, features = denseToSparse(histogram)
    return vis(locations, features, model_input_size, labels, path_name=path_name)

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = torch.nonzero(torch.abs(dense_tensor).sum(axis=-1))
    locations = torch.cat((non_zero_indices[:, 1:], non_zero_indices[:, 0, None]), dim=-1)

    select_indices = non_zero_indices.split(1, dim=1)
    features = torch.squeeze(dense_tensor[select_indices], dim=-2)

    return locations, features


def vis(locations, features, model_input_size, labels, save_num=9, path_name=None):
    # try:
    #     import visualizations
    # except:
    #     import utils.visualizations as visualizations
    import dataset.visualizations as visualizations
    save_num = min(len(labels), save_num)
    h = 1 #int(math.sqrt(save_num))+1
    w = 1 #int(save_num / h)
    images = []
    raw_images = []
    for idx in range(save_num):
        batch_one_mask = locations[:, -1] == idx
        vis_locations = locations[batch_one_mask, :2]
        feature = features[batch_one_mask, :]
        image = visualizations.visualizeLocations(vis_locations.cpu().int().numpy(), model_input_size,
                                                  features=feature.cpu().numpy(), path_name=path_name)
        raw_images.append(image)
        if labels[idx]:
            tmp_img = np.full((30, image.shape[1], 3), 128, dtype="uint8")
            cv2.putText(tmp_img, labels[idx], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 0, 255), 2)
            image = np.vstack((image, tmp_img))
            block_img = np.full((image.shape[0], 20, 3), 128, dtype="uint8")
            image = np.hstack((image, block_img))

        images.append(image)
    if path_name:
        return
    return merge_vis(images, (h, w)), raw_images


def merge_vis(image_ls, size):
    h, w = size
    tmp_images = []
    for idx in range(h):
        tmp_images.append(np.concatenate(image_ls[idx*h: idx *h+ w], axis=1))
    return np.concatenate(tmp_images, axis=0)




