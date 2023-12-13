
from matplotlib import pyplot as plt
from sparsity import *

fig = plt.figure()


colors = ["#145984", "#03ba0c", "#4f4f4f"]
line_width = 4.2
text_size = 16
marker_size = 100
linestyles = ['--', '-.', '-']
legend_size = 12
title_size = 15
x_label_size = 15
y_label_size = 15

dataset_names = ["N-Caltech101", 'DvsGesture', "ASL-DVS", "N-MNIST", "RoShamBo17"]


def plot_DVS():
    plt.subplot(1, 5, 2)
    acc = [94.3, 95.3]
    x_offset = [-0.1, -0.1, -0.1]
    y_offset = [-0.02, 0.1, -0.02]

    name = ["input", "r1", "r2", "r3", "r4", "r5"]
    x_coord = range(len(name))

    dense = dvs_standard_sparsity
    mink_original = dvs_mink_sparsity
    mink_pruned = dvs_mink_drop_sparsity

    plt.plot(x_coord, dense, label='standard conv({}%)'.format(acc[0]), color=colors[0], linewidth=line_width,
             markersize=marker_size, linestyle=linestyles[0], zorder=5)
    plt.plot(x_coord, mink_original, label='submanifold conv({}%)'.format(acc[1]), color=colors[1],
             linewidth=line_width, markersize=marker_size, linestyle=linestyles[1], zorder=4)
    # plt.plot(x_coord, mink_pruned, label='submanifold \nconv + dynamic \npruning'.format(acc[-1]), color=colors[2],
    #          linewidth=line_width, markersize=marker_size, linestyle=linestyles[2], zorder=3)
    #
    # plt.text(len(x_coord) - 1 - x_offset[0], dense[-1] - y_offset[0], str(acc[0]), fontsize=text_size, color=colors[0], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[1], mink_original[-1] - y_offset[1], str(acc[1]), fontsize=text_size, color=colors[1], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[2], mink_pruned[-1] - y_offset[2], str(acc[2]), fontsize=text_size, color=colors[2], weight="bold")

    # plt.xlabel('Resolution', fontsize=12, fontweight=True)
    plt.xticks(x_coord, name, fontweight=True, weight="bold", fontsize=x_label_size)
    plt.xlim(-0.4, len(x_coord)-0.5)
    plt.ylim(0, 1.1)
    plt.yticks(weight="bold", fontsize=y_label_size)
    plt.grid(True, lw=.3, ls='--', c='black', alpha=0.3)
    plt.tick_params(direction='in')
    plt.title(dataset_names[1], fontsize=title_size, weight='bold')

    plt.legend(loc="lower right", prop={"weight": "bold", "size": legend_size}).set_zorder(2)
    # plt.legend(bbox_to_anchor=(0.43, 0.96), prop={"weight": "bold", "size": 12})
    # plt.tight_layout()
    # plt.show()


def plot_NCal():
    plt.subplot(1, 5, 1)
    acc = [73.3, 75.5]
    x_offset = [-0.1, -0.1, -0.1]
    y_offset = [-0.02, -0.02, -0.02]

    dense = ncal_standard_sparsity
    mink_original = ncal_mink_sparsity
    mink_pruned = ncal_mink_drop_sparsity

    name = ["input", "r1", "r2", "r3", "r4", "r5"]
    x_coord = range(len(name))

    plt.plot(x_coord, dense, label='standard conv({}%)'.format(acc[0]), color=colors[0], linewidth=line_width,
             markersize=marker_size, linestyle=linestyles[0], zorder=5)
    plt.plot(x_coord, mink_original, label='submanifold conv({}%)'.format(acc[1]), color=colors[1],
             linewidth=line_width, markersize=marker_size, linestyle=linestyles[1],  zorder=4)
    # plt.plot(x_coord, mink_pruned, label='submanifold \nconv + dynamic \npruning'.format(acc[-1]), color=colors[2],
    #          linewidth=line_width, markersize=marker_size, linestyle=linestyles[2],  zorder=3)
    #
    # plt.text(len(x_coord) - 1 - x_offset[0], dense[-1] - y_offset[0], str(acc[0]), fontsize=text_size, color=colors[0], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[1], mink_original[-1] - y_offset[1], str(acc[1]), fontsize=text_size, color=colors[1], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[2], mink_pruned[-1] - y_offset[2], str(acc[2]), fontsize=text_size, color=colors[2], weight="bold")

    # plt.ylabel('Sparsity ratio', fontsize=y_label_size, weight='bold')
    plt.xlim(-0.4, len(x_coord)-0.5)
    plt.ylabel('Spatial Sparsity', fontsize=y_label_size, weight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(x_coord, name, fontweight=True, weight="bold", fontsize=x_label_size)
    plt.yticks(weight="bold", fontsize=y_label_size)
    plt.grid(True, lw=.3, ls='--', c='black', alpha=0.3)
    plt.tick_params(direction='in')
    plt.legend(loc="lower right", prop={"weight": "bold", "size": legend_size-0.3}).set_zorder(2)
    plt.title(dataset_names[0], fontsize=title_size, weight='bold')

    # plt.legend(bbox_to_anchor=(0.43, 0.96), prop={"weight": "bold", "size": 12})


def plot_ASL():
    plt.subplot(1, 5, 3)
    acc = [99.5, 99.6]
    x_offset = [-0.1, -0.1, -0.1]
    y_offset = [-0.02, -0.02, -0.02]

    dense = asl_standard_sparsity
    mink_original = asl_mink_sparsity
    mink_pruned = asl_mink_drop_sparsity

    name = ["input", "r1", "r2", "r3", "r4", "r5"]
    x_coord = range(len(name))

    plt.plot(x_coord, dense, label='standard conv({}%)'.format(acc[0]), color=colors[0], linewidth=line_width,
             markersize=marker_size, linestyle=linestyles[0], zorder=5)
    plt.plot(x_coord, mink_original, label='submanifold conv({}%)'.format(acc[1]), color=colors[1],
             linewidth=line_width, markersize=marker_size, linestyle=linestyles[1],  zorder=4)
    # plt.plot(x_coord, mink_pruned, label='submanifold \nconv + dynamic \npruning'.format(acc[-1]), color=colors[2],
    #          linewidth=line_width, markersize=marker_size, linestyle=linestyles[2],  zorder=3)
    #
    # plt.text(len(x_coord) - 1 - x_offset[0], dense[-1] - y_offset[0], str(acc[0]), fontsize=text_size, color=colors[0], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[1], mink_original[-1] - y_offset[1], str(acc[1]), fontsize=text_size, color=colors[1], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[2], mink_pruned[-1] - y_offset[2], str(acc[2]), fontsize=text_size, color=colors[2], weight="bold")

    # plt.ylabel('Sparsity ratio', fontsize=y_label_size, weight='bold')
    plt.xlim(-0.4, len(x_coord)-0.5)
    plt.ylim(0, 1.1)
    plt.xticks(x_coord, name, fontweight=True, weight="bold", fontsize=x_label_size)
    plt.yticks(weight="bold", fontsize=y_label_size)
    plt.grid(True, lw=.3, ls='--', c='black', alpha=0.3)
    plt.tick_params(direction='in')
    plt.legend(loc="upper left", prop={"weight": "bold", "size": legend_size}).set_zorder(2)
    plt.title(dataset_names[2], fontsize=title_size, weight='bold')


def plot_NMNIST():
    plt.subplot(1, 5, 4)
    acc = [99.0, 98.8]
    x_offset = [-0.1, -0.1, -0.1]
    y_offset = [-0.02, -0.02, -0.02]

    dense = nmnist_standard_sparsity
    mink_original = nmnist_mink_sparsity
    mink_pruned = nmnist_mink_drop_sparsity

    name = ["input", "r1", "r2", "r3"]
    x_coord = range(len(name))

    plt.plot(x_coord, dense, label='standard conv({}%)'.format(acc[0]), color=colors[0], linewidth=line_width,
             markersize=marker_size, linestyle=linestyles[0], zorder=5)
    plt.plot(x_coord, mink_original, label='submanifold conv({}%)'.format(acc[1]), color=colors[1],
             linewidth=line_width, markersize=marker_size, linestyle=linestyles[1],  zorder=4)
    # plt.plot(x_coord, mink_pruned, label='submanifold \nconv + dynamic \npruning'.format(acc[-1]), color=colors[2],
    #          linewidth=line_width, markersize=marker_size, linestyle=linestyles[2],  zorder=3)
    #
    # plt.text(len(x_coord) - 1 - x_offset[0], dense[-1] - y_offset[0], str(acc[0]), fontsize=text_size, color=colors[0], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[1], mink_original[-1] - y_offset[1], str(acc[1]), fontsize=text_size, color=colors[1], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[2], mink_pruned[-1] - y_offset[2], str(acc[2]), fontsize=text_size, color=colors[2], weight="bold")

    # plt.ylabel('Sparsity ratio', fontsize=y_label_size, weight='bold')
    plt.xticks(x_coord, name, fontweight=True, weight="bold", fontsize=x_label_size)
    plt.yticks(weight="bold", fontsize=y_label_size)
    plt.xlim(-0.3, len(x_coord)-0.5)
    plt.ylim(0, 1.1)
    plt.grid(True, lw=.3, ls='--', c='black', alpha=0.3)
    plt.tick_params(direction='in')
    plt.legend(loc="lower right", prop={"weight": "bold", "size": legend_size}).set_zorder(2)
    plt.title(dataset_names[3], fontsize=title_size, weight='bold')


def plot_Rosh():
    plt.subplot(1, 5, 5)
    acc = [99.4, 99.4]
    x_offset = [-0.1, -0.1, -0.1]
    y_offset = [-0.02, -0.02, -0.02]

    dense = inirosh_standard_sparsity
    mink_original = inirosh_mink_sparsity
    mink_pruned = inirosh_mink_drop_sparsity

    name = ["input", "r1", "r2", "r3", "r4"]
    x_coord = range(len(name))

    plt.plot(x_coord, dense, label='standard conv({}%)'.format(acc[0]), color=colors[0], linewidth=line_width,
             markersize=marker_size, linestyle=linestyles[0], zorder=5)
    plt.plot(x_coord, mink_original, label='submanifold conv({}%)'.format(acc[1]), color=colors[1],
             linewidth=line_width, markersize=marker_size, linestyle=linestyles[1], zorder=4)
    # plt.plot(x_coord, mink_pruned, label='submanifold \nconv + dynamic \npruning'.format(acc[-1]), color=colors[2],
    #          linewidth=line_width, markersize=marker_size, linestyle=linestyles[2], zorder=3)

    # plt.text(len(x_coord) - 1 - x_offset[0], dense[-1] - y_offset[0], str(acc[0]), fontsize=text_size, color=colors[0], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[1], mink_original[-1] - y_offset[1], str(acc[1]), fontsize=text_size, color=colors[1], weight="bold")
    # plt.text(len(x_coord) - 1 - x_offset[2], mink_pruned[-1] - y_offset[2], str(acc[2]), fontsize=text_size, color=colors[2], weight="bold")

    # plt.ylabel('Sparsity ratio', fontsize=y_label_size, weight='bold')
    plt.xticks(x_coord, name, fontweight=True, weight="bold", fontsize=x_label_size)
    plt.xlim(-0.3, len(x_coord)-0.5)
    plt.ylim(0, 1.1)
    plt.yticks(weight="bold", fontsize=y_label_size)
    plt.grid(True, lw=.3, ls='--', c='black', alpha=0.3)
    plt.tick_params(direction='in')
    plt.legend(loc="lower right", prop={"weight": "bold", "size": legend_size}).set_zorder(2)
    plt.title(dataset_names[4], fontsize=title_size, weight='bold')


plt.figure(figsize=(30, 4))
plot_NCal()
plot_DVS()
plot_ASL()
plot_NMNIST()
plot_Rosh()
plt.savefig("Sparsity.svg", format='svg')
plt.show()

