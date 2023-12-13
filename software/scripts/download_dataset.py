import tonic
from tonic import SlicedDataset
from tonic.slicers import SliceByTime

slicing_time_window = 500000
root_path = "/vol/datastore/event_dataset/DVS_slice/"

dataset = tonic.datasets.DVSGesture(save_to=root_path)
slicer = SliceByTime(time_window=slicing_time_window)
sliced_dataset = SlicedDataset(
    dataset, slicer=slicer, metadata_path=root_path + "training_slice.h5"
)

dataset = tonic.datasets.DVSGesture(save_to=root_path, train=False)
slicer = SliceByTime(time_window=slicing_time_window)
sliced_dataset = SlicedDataset(
    dataset, slicer=slicer, metadata_path=root_path + "validation_slice.h5"
)

sensor_size = tonic.datasets.DVSGesture.sensor_size

