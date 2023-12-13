import matplotlib.pyplot as plt

title = "NCal"


# Define the data
values = [74.49, 75.67, 74.11, 70.67, 0, 74.96, 72.28]

# Define the x-axis labels
labels = ['Standard', 'SBMF', 'SBMF \n absSum', 'SBMF \n absSum \n Quant', "", 'SBMF \n random', 'SBMF \n random \nQuant']

# Define the colors for each bar
colors = ['red', 'orange', 'yellow', 'green', "black", 'blue', 'purple']

# Create the bar graph
plt.bar(labels, values, color=colors)

# Add a title and axis labels
plt.title(title, fontsize=20, fontweight='bold')
# plt.xlabel('X-axis Label')
plt.ylabel('Top-1', fontsize=14, fontweight='bold')
plt.ylim(60, 80)
plt.yticks(weight="bold", fontsize=15)
plt.xticks(weight="bold", fontsize=10)
plt.tight_layout()
# Show the graph
plt.show()