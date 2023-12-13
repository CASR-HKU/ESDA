import matplotlib.pyplot as plt
import numpy as np

# Define the data for the plot
data = np.array([[5.968184043, 3.140367445, 1.540168421, 0.701069457, 1],
                 [3.875942549, 2.299882817, 1.439554578, 0.957341138, 1],
                 [9.000938086, 5.231733915, 2.565050793, 1.083120108, 1],
                 [2.863800905, 1.966749534, 1.355246253, 0.927733802, 1],
                 [2.863604732, 2.351428571, 1.591260634, 0.876651044, 1],
                 [2.854050279, 2.356978085, 1.579822188, 0.862236287, 1],
                 [2.108910891, 2.009433962, 1.450953678, 1.140664049, 1],
                 [2.689453125, 2.058295964, 1.675182482, 1.099840256, 1],
                 [2.783864542, 2.056659308, 1.797427653, 1.113545817, 1],
                 [2.713864307, 2.076749436, 1.708978328, 1.095672886, 1],
                 [2.743564356, 2.015272727, 1.752688172, 1.158444816, 1],
                 [2.649621212, 2.057352941, 1.746566792, 1.118305356, 1],
                 [2.694230769, 2.015827338, 1.819480519, 1.099686028, 1],
                 [1.882237488, 1.453030303, 1.330097087, 1.003663004, 1],
                 [2.220630372, 1.982097187, 1.658345221, 1.247317597, 1],
                 [2.309734513, 1.950996678, 1.638075314, 1.259517426, 1],
                 [2.331702544, 2.110717449, 1.659470752, 1.298637602, 1]])

# Extract the data values and labels for each group
# values = data[:, :-1]
blank_length = 2
bar_labels = ["Sparse 0.1", "Sparse 0.2", "Sparse 0.4", "Sparse 0.8", "Dense"]
labels = ["block_{}".format(i) for i in range(len(data))]
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

