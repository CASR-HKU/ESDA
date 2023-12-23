import numpy as np
import os
from .transform_func import Denoiser
import h5py
import scipy.io as scio


def write_labels(path, labels):
    path = "/".join(path.split("/")[:-1]) + "/labels.txt"
    with open(path, "w") as f:
        for label in labels:
            f.write(label + "\n")


def preprocess_event(dataset, window, overlap_ratio, save_folder="", vis_hist=False, original_format=False,
                     denoise_time=50000
                     ):
    window = int(window)
    step = int(window * (1-overlap_ratio))
    assert step > 0
    # dataset_name = dataset.dataset_name
    files, labels = dataset.dataset.files, dataset.dataset.labels
    denoise_op = dataset.transform.transforms[0]
    raw_length, denoise_length, time_elapse = [], [], []

    # assert isinstance(denoise_op, Denoiser) and len(dataset.transform.transforms) <= 1, \
    #     "Other augmentations should not be included in preprocessing"
    # assert save_folder or vis_hist, "Please select file-saving or visualization mode"

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        write_labels(os.path.join(save_folder, "labels.txt"), dataset.dataset.object_classes)

    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print("Processing sample {}".format(idx))

        events, label = dataset.load_sample(idx)
        file = files[idx]
        file_name = file.split("/")[-1].split(".")[0]

        time_max = len(events)
        t_begin = 0
        while t_begin < time_max:
            if original_format:
                clipped_events = events
            else:
                if len(events) < window:
                    clipped_events = events
                else:
                    clipped_events = events[t_begin: t_begin+window]
                    if len(clipped_events) < window:
                        break

            time_elapse.append(clipped_events[-1][2] - clipped_events[0][2])
            denoise_clipped_events = denoise_op(clipped_events, filter_time=denoise_time)
            raw_length.append(clipped_events.shape[0])
            denoise_length.append(denoise_clipped_events.shape[0])
            if original_format:
                t_begin += time_max
            else:
                t_begin += step

            if save_folder:
                if original_format:
                    os.makedirs(os.path.join(save_folder, "/".join(file.split("/")[-3:-1])), exist_ok=True)
                    target_name = os.path.join(save_folder, "/".join(file.split("/")[-3:]))
                    np.save(target_name, denoise_clipped_events)
                else:
                    target_name = os.path.join(save_folder, "{}-begin_{}".format(file_name, t_begin))
                    content = {
                        "event": clipped_events,
                        "denoise_event": denoise_clipped_events,
                        "label": label
                    }
                    np.save(target_name+".npy", content)

    raw_ave = sum(raw_length) / len(raw_length)
    denoise_ave = sum(denoise_length) / len(denoise_length)
    print("Time elapse: {}".format(sum(time_elapse) / len(time_elapse)))
    print("Ratio: {}".format(denoise_ave / raw_ave))


def preprocess_time(dataset, window, overlap_ratio, save_folder="", vis_hist=False, original_format=False,
                    denoise_time=None):
    step = window * (1-overlap_ratio)
    assert step > 0
    # dataset_name = dataset.dataset_name
    files, labels = dataset.dataset.files, dataset.dataset.labels
    denoise_op = dataset.transform.transforms[0]

    # Only keep sampling
    # dataset.transform.transforms = [dataset.transform.transforms[1]]
    # assert isinstance(denoise_op, Denoiser) and len(dataset.transform.transforms) <= 1, \
    #     "Other augmentations should not be included in preprocessing"
    # assert save_folder or vis_hist, "Please select file-saving or visualization mode"

    if save_folder:
        os.makedirs("/".join(save_folder.split("/")[:-1]), exist_ok=True)
        f = h5py.File(save_folder, "w")

    raw_length, denoise_length = [], []

    for idx in range(len(dataset)):
        if idx % 100 == 0:
            print("Processing sample {}".format(idx))
            if raw_length:
                raw_ave = sum(raw_length) / len(raw_length)
                denoise_ave = sum(denoise_length) / len(denoise_length)
                # print("Ratio: {}".format(denoise_ave / raw_ave))

        events, label = dataset.load_sample(idx)
        file = files[idx]
        file_name = file.split("/")[-1].split(".")[0]

        time_max = events[:, 2].max() - window
        t_begin = min(0, time_max) - 1e-8
        while t_begin < time_max:
            if original_format:
                clipped_events = events
            else:
                begin_idx = np.where(events[:, 2] > t_begin)[0]
                end_idx = np.where(events[:,2] < t_begin+window)[0]
                clipped_events = events[np.intersect1d(begin_idx, end_idx), ...]

            denoise_clipped_events = denoise_op(clipped_events)
            raw_length.append(clipped_events.shape[0])
            denoise_length.append(denoise_clipped_events.shape[0])

            if original_format:
                t_begin += time_max
            else:
                t_begin += step

            if save_folder:
                if len(denoise_clipped_events) > 0:
                    if original_format:
                        os.makedirs(os.path.join(save_folder, "/".join(file.split("/")[-3:-1])), exist_ok=True)
                        target_name = os.path.join(save_folder, "/".join(file.split("/")[-3:]))
                        np.save(target_name, denoise_clipped_events)
                    else:
                        if time_max < 100:
                            my_group = f.create_group("{}-t{}".format(file_name, int(t_begin*1000)))
                        else:
                            my_group = f.create_group("{}-t{}".format(file_name, int(t_begin)))
                        my_group.create_dataset("label", data=np.array(label))
                        my_group.create_dataset("data", data=denoise_clipped_events)

    raw_ave = sum(raw_length)/len(raw_length)
    denoise_ave = sum(denoise_length)/len(denoise_length)
    print("Ratio: {}".format(denoise_ave/raw_ave))
