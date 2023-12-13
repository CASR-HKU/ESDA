import matplotlib.pyplot as plt
import numpy as np


def cal_speedup(target, base):
    return [base[i] / target[i] for i in range(len(target))]


# Define the data for the plot
jetson_cpu = [0.1048, 0.1048, 0.1048, 0.1048, 0.1048]
jetson_gpu = [0.0672, 0.0482, 0.0518, 0.569, 0.447]
ours = [0.002, 0.002, 0.002, 0.002, 0.002]


CPU_data = cal_speedup(jetson_cpu, jetson_cpu)
GPU_data = cal_speedup(jetson_gpu, jetson_cpu)
ours_data = cal_speedup(ours, jetson_cpu)

# Extract the data values and labels for each group
# values = data[:, :-1]
blank_length = 1
bar_labels = ['DvsGesture', "N-Caltech101", "ASL-DVS", "N-MNIST", "POKER-DVS"]
# labels = ["block_{}".format(i) for i in range(len(data))]
legend_size = 15

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 4))
max_height = 4

# Set the bar width
bar_width = 0.1

# Set the positions of the x-axis ticks and labels
# x_pos = np.arange(len(data[0]))

# Specify the colors of the bars
colors = ["#af1626", '#9b9913', '#15a62f', '#14439f', '#626262']
labels_x = []

# Iterate over each group of bars
for i in range(len(data)):
    # Calculate the position of the left side of the bars
    left = i * (len(data[0])+blank_length) * bar_width
    labels_x.append(left-bar_width*1)

    # Create a bar chart for the group
    for j in range(len(data[0])):
        ax.bar(left + (bar_width*j)*0.9, data[i][j], width=bar_width, color=colors[j])
        if data[i][j] > max_height:
            under_height = 1 if j % 2 == 0 else 1.2
            plt.text(left + (bar_width*j)*0.9, max_height - under_height, "{:.0f}".format(data[i][j]),
                         ha='center', va='bottom', fontsize=12, weight="bold")


ax.set_ylabel('Speed up', fontweight='bold', fontsize=14)
ax.set_title('Speed up of different blocks', fontweight='bold', fontsize=16)
plt.yticks(weight="bold", fontsize=15)
# plt.xticks(weight="bold", fontsize=15)
plt.xticks(labels_x, labels, weight="bold", fontsize=15, rotation=45)
ax.axes.xaxis.set_visible(False)

# Set the y-axis limits
ax.set_ylim([0, max_height])
plt.tight_layout()
ax.legend(bar_labels,  fontsize=14, prop={"weight": "bold", "size": legend_size}, ncol=3)
plt.tick_params(direction='in')
plt.show()
plt.savefig("SpeedUp.pdf")

# Show the plot

