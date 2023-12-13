import numpy as np
from typing import BinaryIO, Optional, Union

events_struct = np.dtype(
    [("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)]
)


def read_aedat4(in_file):
    """Get the aer events from version 4 of .aedat file.

    Parameters:
        in_file: str The name of the .aedat file

    Returns:
        events:   numpy structured array of events
    """
    import aedat

    decoder = aedat.Decoder(in_file)
    target_id = None
    width = None
    height = None
    for stream_id, stream in decoder.id_to_stream().items():
        if stream["type"] == "events" and (target_id is None or stream_id < target_id):
            target_id = stream_id
    if target_id is None:
        raise Exception("there are no events in the AEDAT file")
    parsed_file = {
        "type": "dvs",
        "width": width,
        "height": height,
        "events": np.concatenate(
            tuple(
                packet["events"]
                for packet in decoder
                if packet["stream_id"] == target_id
            )
        ),
    }
    return parsed_file["events"]


def make_structured_array(*args, dtype=events_struct):
    """Make a structured array given a variable number of argument values.

    Parameters:
        *args: Values in the form of nested lists or tuples or numpy arrays.
               Every except the first argument can be of a primitive data type like int or float.

    Returns:
        struct_arr: numpy structured array with the shape of the first argument
    """
    assert not isinstance(
        args[-1], np.dtype
    ), "The `dtype` must be provided as a keyword argument."
    names = dtype.names
    assert len(args) == len(names)
    struct_arr = np.empty_like(args[0], dtype=dtype)
    for arg, name in zip(args, names):
        struct_arr[name] = arg
    return struct_arr


def read_mnist_file(
    bin_file: Union[str, BinaryIO], dtype: np.dtype = events_struct, is_stream: Optional[bool] = False
):
    """Reads the events contained in N-MNIST/N-CALTECH101 datasets.

    Code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
    """
    if is_stream:
        raw_data = np.frombuffer(bin_file.read(), dtype=np.uint8).astype(np.uint32)
    else:
        with open(bin_file, "rb") as fp:
            raw_data = np.fromfile(fp, dtype=np.uint8).astype(np.uint32)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

    # Process time stamp overflow events
    time_increment = 2**13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    # Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    xytp = make_structured_array(
        all_x[td_indices],
        all_y[td_indices],
        all_ts[td_indices],
        all_p[td_indices],
        dtype=dtype,
    )
    return xytp