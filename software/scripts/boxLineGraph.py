

file_path = "sparsity_DVS_save.txt"

import matplotlib.pyplot as plt
import numpy as np

with open(file_path, "r") as f:
    data = f.readlines()[1:]
    data = [list(map(lambda x:float(x), d[:-1].split("\t")[2:])) for d in data]
# data = np.loadtxt(file_path)[1:, 2:]
# all_data = [np.random.normal(0,std,100) for std in range(1,4)]

figure, axes = plt.subplots()
# axes.boxplot(data, patch_artist=True)
axes.boxplot(np.transpose(np.array(data)).tolist(), patch_artist=True)
plt.title("Sparsity of DVS dataset")
plt.show()

