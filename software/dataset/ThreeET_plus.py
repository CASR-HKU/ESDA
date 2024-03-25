import os
from typing import Any, Callable, Optional, Tuple
import h5py
import numpy as np

from tonic.dataset import Dataset

class ThreeETplus_Eyetracking(Dataset):
    """3ET DVS eye tracking `3ET <https://github.com/qinche106/cb-convlstm-eyetracking>`_
    ::

        @article{chen20233et,
            title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
            author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
            journal={arXiv preprint arXiv:2308.11771},
            year={2023}
        }

        authors: Qinyu Chen^{1,2}, Zuowen Wang^{1}
        affiliations: 1. Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland
                      2. Univeristy of Leiden, Netherlands

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.

    Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        data_dir = save_to
        data_list_dir = './dataset'
        # Load filenames from the provided lists
        if split == "train":
            filenames = self.load_filenames(os.path.join(data_list_dir, "train_files.txt"))
        elif split == "val":
            filenames = self.load_filenames(os.path.join(data_list_dir, "val_files.txt"))
        elif split == "test":
            filenames = self.load_filenames(os.path.join(data_list_dir, "test_files.txt"))
        else:
            raise ValueError("Invalid split name")

        # Get the data file paths and target file paths
        if split == "train" or split == "val":
            self.data = [os.path.join(data_dir, "train", f, f + ".h5") for f in filenames]
            self.targets = [os.path.join(data_dir, "train", f, "label.txt") for f in filenames]
        elif split == "test":
            self.data = [os.path.join(data_dir, "test", f, f + ".h5") for f in filenames]
            # for test set, we load the placeholder labels with all zeros
            self.targets = [os.path.join(data_dir, "test", f, "label_zeros.txt") for f in filenames]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # get events from .h5 file
        with h5py.File(self.data[index], "r") as f:
            # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
            # t is in us
            events = f["events"][:].astype(self.dtype)
            events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
            
        # load the sparse labels
        with open(self.targets[index], "r") as f:
            # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
            target = np.array(
                [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    def load_filenames(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]