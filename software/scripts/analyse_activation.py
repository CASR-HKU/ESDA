import numpy as np
import os
from collections import defaultdict

src_folder = "activation"
npy_name = os.listdir(os.path.join(src_folder, os.listdir(src_folder)[0]))
npy_name.sort()
keys = set(["_".join(name.split("_")[:-1]) for name in npy_name])

mean, std = defaultdict(list), defaultdict(list)
for key in keys:
    for folder in os.listdir(src_folder):
        if "sum" in folder:
            continue
        sub_folder = os.path.join(src_folder, folder)
        mean_file = os.path.join(sub_folder, key + "_mean.npy")
        std_file = os.path.join(sub_folder, key + "_std.npy")
        mean[key].append(np.load(mean_file))
        std[key].append(np.load(std_file))

for key in keys:
    mean[key] = np.array(mean[key]).mean(axis=0)
    std[key] = np.array(std[key]).mean(axis=0)

for key in keys:
    os.makedirs(os.path.join(src_folder, "sum"), exist_ok=True)
    np.savetxt(os.path.join(src_folder, "sum", key + "_mean.txt"), mean[key])
    np.savetxt(os.path.join(src_folder, "sum", key + "_std.txt"), std[key])

a = 1

