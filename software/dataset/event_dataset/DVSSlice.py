import tonic
from tonic import SlicedDataset
from tonic.slicers import SliceByTime
import numpy as np
from scipy import ndimage


sensor_size = tonic.datasets.DVSGesture.sensor_size


class DVSSlicedDataset:
    def __init__(self, root, mode="training", shuffle=False, slicing_time_window=500000, drop_event_ratio=0.2,
                 drop_area_ratio=0.1, overlap=0, **kwargs):
        is_train = True if mode == "training" else False
        meta_file_name = "slice_metadata_training.h5" if is_train else "slice_metadata_validation.h5"
        self.istrain = is_train
        dataset = tonic.datasets.DVSGesture(save_to=root, train=is_train)
        slicing_time_window = slicing_time_window  # microseconds
        self.object_classes = dataset.classes
        slicer = SliceByTime(time_window=slicing_time_window, overlap=overlap)
        meta_name = "{}_slice.h5".format(mode) if overlap == 0 else "{}_slice_0p{}_overlap.h5".format(mode, overlap*100)
        sliced_dataset = SlicedDataset(
            dataset, slicer=slicer, metadata_path=root + "_win{}w/".format(int(slicing_time_window/10000)) + meta_file_name
        )

        if is_train:
            transforms = tonic.transforms.Compose(
                # rotate,
                [
                    tonic.transforms.DropEvent(p=drop_event_ratio),
                    tonic.transforms.DropPixel(
                        coordinates=[
                            [x, y]
                            for x in np.random.randint(128, size=10)
                            for y in np.random.randint(128, size=10)
                        ]
                    ),
                    tonic.transforms.DropEventByArea(
                        sensor_size=sensor_size, area_ratio=drop_area_ratio
                    ),
                    tonic.transforms.ToImage(
                        sensor_size=tonic.datasets.DVSGesture.sensor_size
                    )
                ]
            )
        else:
            transforms = tonic.transforms.Compose(
                # rotate,
                [
                    tonic.transforms.ToImage(
                        sensor_size=tonic.datasets.DVSGesture.sensor_size
                    )
                ]
            )

        self.sliced_dataset = SlicedDataset(
            dataset, slicer=slicer, transform=transforms, metadata_path=root + "_win{}w/".format(int(slicing_time_window/10000)) + meta_file_name
        )
        self.files = self.sliced_dataset.slice_dataset_map
        self.nr_classes = len(self.object_classes)

    def __getitem__(self, idx):
        data, label = self.sliced_dataset[idx]
        if self.istrain:
            data = self.random_shift(data)
            data = self.rotate(data)
        return data, label

    def rotate(self, input_array):
        rotation_angle = np.random.uniform(-20, 20)  # Generate a random rotation angle between -10 and 10 degrees
        rotated_array = ndimage.rotate(input_array, rotation_angle, axes=(1, 2), reshape=False, order=1)
        return rotated_array

    def random_shift_crop(self, input_image):
        padded_image = np.pad(input_image, ((0, 0), (4, 4), (4, 4)), mode='constant')

        _, h, w = padded_image.shape

        x = np.random.randint(0, w - 63)
        y = np.random.randint(0, h - 63)

        cropped_image = padded_image[:, x:x + 64, y:y + 64]  # Crop a (128, 128) image

        return cropped_image

    def random_shift(self, input_image):
        padded_image = np.pad(input_image, ((0, 0), (10, 10), (10, 10)), mode='constant')  # Pad the image with 10 pixels

        if np.random.rand() < 0.5:
            x = np.random.randint(0, 21)  # Generate random x-coordinate for cropping
            y = np.random.randint(0, 21)  # Generate random y-coordinate for cropping

            cropped_image = padded_image[:, x:x + 128, y:y + 128]  # Crop a (128, 128) image
        else:
            cropped_image = padded_image[:, 10:138, 10:138]  # No crop, return the center (128, 128) image

        return cropped_image