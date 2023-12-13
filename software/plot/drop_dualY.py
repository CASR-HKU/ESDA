import matplotlib.pyplot as plt
import numpy as np

# Sample data
ratio = [0.05, 0.1, 0.15, 0.2]
accuracy = [0.95, 0.94, 0.93, 0.91]
latency = [1.2, 1.5, 2, 3]
linestyle = ['-', '-']
label_size = 12

# Create figure and axis objects
fig, ax1 = plt.subplots()

# Plot the first y-axis data
ax1.plot(ratio, accuracy, 'o-', color='blue', label='Accuracy', linestyle=linestyle[0])

# Set the x and y-axis labels
# ax1.set_xlabel('Ratio')
ax1.set_ylabel('Accuracy', fontweight='bold')

# Set the y-axis ticks and limits
ax1.tick_params(axis='y')
ax1.set_ylim(0.9, 0.96)
ax1.set_yticks(np.arange(0.9, 0.97, 0.02))

# Create a second y-axis object
ax2 = ax1.twinx()

# Plot the second y-axis data
ax2.plot(ratio, latency, 'o-', color='red', label='Latency', linestyle=linestyle[1])

# Set the y-axis label for the second axis
ax2.set_ylabel('Latency (ms)', fontweight='bold')

# Set the y-axis ticks and limits for the second axis
ax2.tick_params(axis='y', )
ax2.set_ylim(1, 3.2)
ax2.set_yticks(np.arange(1, 3.2, 0.4))


# Set the x-axis ticks and limits
ax1.set_xticks(ratio)
ax1.set_xlim(min(ratio), max(ratio))
ax1.tick_params(direction='in')
ax2.tick_params(direction='in')
plt.xlim(0.025, 0.225)
plt.grid(True)

for tick in ax1.get_yticklabels():
    tick.set_fontsize(label_size)
    tick.set_weight('bold')

for tick in ax1.get_xticklabels():
    tick.set_fontsize(label_size)
    tick.set_weight('bold')

for tick in ax2.get_yticklabels():
    tick.set_fontsize(label_size)
    tick.set_weight('bold')

# Add a legend and title
# ax1.legend(loc='upper left')

plt.tick_params(direction='in')
plt.title('Accuracy vs. Latency with Differnet Pruning Ratio', fontweight='bold')

# Show plot
plt.show()
