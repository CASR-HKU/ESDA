import matplotlib.pyplot as plt
import numpy as np

# Sample data
cycles = np.array([3763.927677, 5597.881871, 2799.704626, 3064.870809, 5597.791959, 5597.848733, 5597.728195, 5335.93318, 3978.947194, 5335.88696, 5335.864077, 5335.756927, 4264.93696, 5597.674297, 5597.903648, 3792.946185, 2843.915393, 5335.909788, 4198.804974, 4456.856294, 2822.895377, 5335.88751, 2822.9511, 5597.91697, 4595.91345, 5335.907209, 2676.92396, 2699.945548, 2876.890941, 2987.905208, 3274.881671, 3655.918528, 3839.79977, 3847.893221, 3963.92396, 4027.873038, 4170.868349, 4243.83834])
accuracy = np.array([95.53, 95.43, 95.31, 95.27, 95.26, 95.06, 95.04, 95, 94.97, 94.93, 94.87, 94.78, 94.77, 94.73, 94.73, 94.6, 94.57, 94.54, 94.48, 94.48, 94.44, 94.38, 94.25, 94.14, 94.11, 93.81, 94.45, 95.1, 94.611, 94.913, 94.381, 94.784, 94.338, 94.798, 95.1, 94.999, 93.792, 95.042])


# Create scatterplot
plt.scatter(cycles, accuracy)

# Add labels and title
plt.title('DVSGesture NAS result', fontweight='bold', fontsize=14, pad=20)
plt.xlabel('Cycles', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy', fontweight='bold', fontsize=12)

# Add gridlines
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

# Make axis labels and ticks bold
plt.xticks(fontweight='bold', fontsize=10)
plt.yticks(fontweight='bold', fontsize=10)
plt.xlim(1500, 6000)
plt.ylim(93.5, 96)

# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)

# Show plot
plt.show()