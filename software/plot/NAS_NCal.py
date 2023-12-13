import matplotlib.pyplot as plt
import numpy as np

# Sample data
cycles = np.array([6466.925313, 8048.896401, 9210.910203, 11969.78062, 12105.9415, 12621.89179, 14991.76224, 14999.90036, 15063.82868, 15116.90076, 15511.88637, 15583.91693, 15831.89011, 15903.78568, 15946.86513, 15959.7429, 15969.8514, 16050.86475, 16353.86539, 16399.8823, 16434.8809, 16434.94087, 16615.86757, 17620.93319])
accuracy = np.array([72.63, 71.997, 73.958, 72.73, 71.21, 69.969, 69.741, 72.32, 71.18, 71.583, 71.74, 73.713, 72.19, 68.625, 68.51, 71.828, 69.808, 73.35, 72.504, 69.94, 71.439, 73.705, 71.465, 72.961])


# Create scatterplot
plt.scatter(cycles, accuracy)

# Add labels and title
plt.title('N-Caltech101 NAS result', fontweight='bold', fontsize=14, pad=20)
plt.xlabel('Cycles', fontweight='bold', fontsize=12)
plt.ylabel('Accuracy', fontweight='bold', fontsize=12)

# Add gridlines
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

# Make axis labels and ticks bold
plt.xticks(fontweight='bold', fontsize=10)
plt.yticks(fontweight='bold', fontsize=10)
plt.xlim(5000, 18000)
plt.ylim(68, 75)

# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(2)
# ax.spines['left'].set_linewidth(2)

# Show plot
plt.show()