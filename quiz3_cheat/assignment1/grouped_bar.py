import numpy as np
import matplotlib.pyplot as plt

# Categories (x-axis labels)
categories = ["Physics", "Math", "Biology"]

# Data for two groups
male = [40, 50, 30]        # values for male students
female = [35, 42, 45]      # values for female students

# Create numerical x positions for each category
x = np.arange(len(categories))

# Width of each bar
width = 0.35

# Create figure and axes
fig, ax = plt.subplots()

# Plot first dataset (shift bars slightly left)
ax.bar(x - width/2, male, width, label="Male", color="LightSkyBlue")

# Plot second dataset (shift bars slightly right)
ax.bar(x + width/2, female, width, label="Female", color="IndianRed")

# Set x-axis tick positions
ax.set_xticks(x)

# Set x-axis labels
ax.set_xticklabels(categories)

# Label the axes
ax.set_xlabel("Department")
ax.set_ylabel("Number of Students")

# Add title
ax.set_title("Students by Department")

# Add legend to distinguish groups
ax.legend()

# Adjust layout so labels fit nicely
plt.tight_layout()

# Display the plot
plt.show()