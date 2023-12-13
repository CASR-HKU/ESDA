import numpy as np
import torch


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
        np.add.at(img_neg, x[p == 0] + self.width * y[p == 0], 1)

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

    # def to_eventQueue(self, events, K=15):
    #     """
    #     Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
    #     coordinates u and v.
    #     """
    #     events = events.astype(np.float32)

    #     if events.shape[0] == 0:
    #         return np.zeros([self.height, self.width, 2*K], dtype=np.float32)

    #     # [2, K, height, width],  [0, ...] time, [:, 0, :, :] newest events
    #     four_d_tensor = er.event_queue_tensor(events, K, self.height, self.width, -1).astype(np.float32)

    #     # Normalize
    #     four_d_tensor[0, ...] = four_d_tensor[0, 0, None, :, :] - four_d_tensor[0, :, :, :]
    #     max_timestep = np.amax(four_d_tensor[0, :, :, :], axis=0, keepdims=True)

    #     # four_d_tensor[0, ...] = np.divide(four_d_tensor[0, ...], max_timestep, where=max_timestep.astype(np.bool))
    #     four_d_tensor[0, ...] = four_d_tensor[0, ...] / (max_timestep + (max_timestep == 0).astype(np.float))

    #     return four_d_tensor.reshape([2*K, self.height, self.width]).transpose(1, 2, 0)

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

