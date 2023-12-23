import tonic
import os


def get_dataset(name, path):
    if name == "DVSGesture":
        dataset = tonic.datasets.DVSGesture
    elif name == "NMNIST":
        dataset = tonic.datasets.NMNIST
    elif name == "NCal":
        dataset = tonic.datasets.NCALTECH101
    elif name == "ASLDVS":
        dataset = tonic.datasets.ASLDVS
    else:
        raise ValueError("Unknown dataset name")

    for istrain in [True, False]:
        target_path = os.path.join(path, "train" if istrain else "test")
        dataset(save_to=target_path, transform=None, train=istrain)
        print("Finishing downloading dataset to {}".format(target_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--path", "-p", type=str, default=".")
    args = parser.parse_args()
    get_dataset(args.dataset, args.path)
