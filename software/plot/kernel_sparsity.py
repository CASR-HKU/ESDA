import matplotlib.pyplot as plt
import numpy as np

dataset_names = ["N-Caltech101", 'DvsGesture', "ASL-DVS", "N-MNIST", "RoShamBo17"]

curve_labels = ["standard conv", "submanifold conv"]#, "submanifold \nconv + dynamic \npruning"]
curve_colors = ["#145984", "#03ba0c"] #, "#4f4f4f"]
linestyles = ['--', '-.'] #, '-']
x_labels = ["input", "r1", "r2", "r3", "r4", "r5"]
line_width = 4.2

x_label_size = 16
y_label_size = 16
legend_size = 13

accuracy = [
    [73.3, 75.5],
    [94.3, 95.3],
    [99.5, 99.6],
    [99.0, 98.8],
    [99.4, 99.4]
]


from sparsity import *

def gen_all_ones(ls):
    return np.ones_like(np.array(ls))


def get_kernel_values(lists):
    return [cal_kernel(ls) for ls in lists]


def cal_kernel(ls):
    return ls[0] * 1/9 + ls[1] * 2/9 + ls[2] * 3/9 + ls[3] * 4/9 + ls[4] * 5/9 + ls[5] * 6/9 + ls[6] * 7/9 + \
        ls[7] * 8/9 + ls[8] * 9/9


data_standard = [get_kernel_values(ncal_standard_kernel), get_kernel_values(dvs_standard_kernel),
               get_kernel_values(asl_standard_kernel), get_kernel_values(nmnist_standard_kernel),
               get_kernel_values(iniRosh_standard_kernel)]
data_mink = [get_kernel_values(ncal_mink_kernel), get_kernel_values(dvs_mink_kernel), get_kernel_values(asl_mink_kernel),
              get_kernel_values(nmnist_mink_kernel), get_kernel_values(iniRosh_mink_kernel)]
data_mink_drop = [get_kernel_values(ncal_mink_drop_kernel), get_kernel_values(dvs_mink_drop_kernel),
              get_kernel_values(asl_mink_drop_kernel), get_kernel_values(nmnist_mink_drop_kernel),
              get_kernel_values(iniRosh_mink_drop_kernel)]

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(30, 4))

for i, dataset in enumerate(dataset_names):
    ax = axs[i]

    ax.set_title(dataset, fontweight='bold', fontsize=14)
    ax.plot(data_standard[i], label="{}({}%)".format(curve_labels[0], accuracy[i][0]),
            color=curve_colors[0], zorder=10, linestyle=linestyles[0], linewidth=line_width)
    ax.plot(data_mink[i], label="{}({}%)".format(curve_labels[1], accuracy[i][1]),
            color=curve_colors[1], zorder=10, linestyle=linestyles[1], linewidth=line_width)
    # ax.plot(data_mink_drop[i], label=curve_labels[2], color=curve_colors[2], zorder=10, linestyle=linestyles[2])

    if i == 0:
        ax.set_ylabel('Kernel Sparsity', fontweight='bold', fontsize=14)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_weight('bold')
    # else:
    #     for tick in ax.get_yticklabels():
    #         tick.set_visible(False)

    ax.set(xticks=range(len(data_standard[i])), xticklabels=x_labels[:len(data_standard[i])])
    ax.set_xlabel("", fontdict={'fontsize': x_label_size, 'fontweight': 'bold'})
    ax.tick_params(direction='in')

    for tick in ax.get_xticklabels():
        tick.set_fontsize(y_label_size)
        tick.set_weight('bold')

    ax.set_ylim([0, 1.1])
    ax.set_xlim([-0.5, len(data_standard[i]) - 0.5])
    ax.legend(loc="lower right", prop={'size': 11, 'weight': 'bold'}, ncol=1).set_zorder(2)
    ax.grid(True, lw=.3, ls='--', c='black', alpha=0.3)


plt.tick_params(direction='in')
plt.savefig('kernel_sparsity.svg', format='svg')
plt.show()
