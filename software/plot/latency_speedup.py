import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Define the datasets and their corresponding nx_gpu and ESDA values
datasets = ["N-Caltech101", 'DvsGesture', "ASL-DVS", "N-MNIST", "RoShamBo17"]
nx_gpu = [48.2, 67.2, 51.8, 56.9, 44.7]
ESDA = [3.13, 0.65, 0.7, 0.15, 0.35]

label_size = 15


def speedup(base, target):
    return [b/t for b, t in zip(base, target)]


speedup_nx = speedup(nx_gpu, nx_gpu)
speedup_esda = speedup(nx_gpu, ESDA)

# Set the width of the bars
bar_width = 0.35

# Create the x-axis values for the bar graph
x1 = np.arange(len(datasets))
x2 = [x + bar_width for x in x1]

# Create the bar graph
fig, ax = plt.subplots()
rects1 = ax.bar(x1, speedup_esda, bar_width, label='ESDA')
rects2 = ax.bar(x2, speedup_nx, bar_width, label='nx_gpu')

# Add labels and titles to the bar graph
font = FontProperties(weight='bold')
# ax.set_xlabel('Dataset', fontproperties=font)
ax.set_ylabel('Speedup', fontproperties=font)
ax.set_title('Latency speedup', fontproperties=font)
ax.set_xticks(x1 + bar_width / 2)
ax.set_xticklabels(datasets)
ax.tick_params(direction='in')

for tick in ax.get_yticklabels():
    tick.set_fontsize(label_size)
    tick.set_weight('bold')

for tick in ax.get_xticklabels():
    tick.set_fontsize(10)
    tick.set_weight('bold')

ax.legend(prop={'size': 11, 'weight': 'bold'}, )

# Display the bar graph
plt.show()
