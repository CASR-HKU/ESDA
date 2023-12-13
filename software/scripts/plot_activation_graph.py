import math
import os
import matplotlib.pyplot as plt

file_quant = "/Users/cheungbh/Downloads/DVS_quant_result/activation.txt"

with open(file_quant, "r") as f:
    lines = f.readlines()
    lines = [line[:-1].split("\t") for line in lines]


folder_pretrain = "/Users/cheungbh/Downloads/activation_baseline"
mean_files = [os.path.join(folder_pretrain, file) for file in os.listdir(folder_pretrain) if "max" in file]
mean_files.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))

mean = []
for mean_file in mean_files:
    with open(mean_file, "r") as f:
        lines_mean = f.readlines()
        lines_mean = [float(line[:-1].split("\t")[0]) for line in lines_mean]
        mean.append(sum(lines_mean)/len(lines_mean))

plt.bar(range(len(mean)), mean)
plt.xlabel("Layer")
plt.ylabel("Mean Activation")
plt.show()

x = range(len(lines[0]))
for idx, line in enumerate(lines[:10]):
    line = [math.log10(float(l)) for l in line]
    plt.plot(x, line, label="layer {}".format(idx))

plt.ylabel('Activation Max (lg)')
plt.xlabel('Iteration')
plt.legend()
plt.show()

a = 1


