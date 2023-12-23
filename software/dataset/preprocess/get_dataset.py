import tonic
import os


def get_dataset(name, path):
    if name == "DVSGesture":
        dataset = tonic.datasets.DVSGesture
    elif name == "NMNIST":
        dataset = tonic.datasets.NMNIST
    elif name == "NCaltech101":
        dataset = tonic.datasets.NCALTECH101
    elif name == "ASLDVS":
        dataset = tonic.datasets.ASLDVS
    else:
        raise ValueError("Unknown dataset name")

    if name in ["ASLDVS", "NCaltech101"]:
        dataset(save_to=path, transform=None)
        print("Finishing downloading {} dataset".format(name))
    else:
        for istrain in [True, False]:
            target = "train" if istrain else "test"
            dataset(save_to=path, transform=None, train=istrain)
            print("Finishing downloading the {} set of {} dataset".format(target, name))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--path", "-p", type=str, default=".")
    args = parser.parse_args()
    get_dataset(args.dataset, args.path)
