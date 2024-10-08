"""
-------------------------------------------------------------------------------
 Title        : significance_analysis.py
 Description  : Calculate statistical difference between 3 groups
 Author       : Guillem Cornella
 Date Created : February 2024
-------------------------------------------------------------------------------
"""

from scipy.stats import ttest_ind, f_oneway
import matplotlib.pyplot as plt
import numpy as np

# p-value calc
young = [0.770, 0.965, 1.046, 1.786, 1.414, 1.094, 1.670, 0.737, 1.267, 1.189, 2.087, 1.005, 1.097, 1.853, 0.914, 1.124, 1.610, 1.821, 1.257, 1.392, 1.260]
older = [1.855, 1.379, 1.303, 1.187, 1.400, 1.618, 0.944, 1.137, 1.042, 1.029, 1.022, 1.829, 1.709, 1.474, 1.336, 1.134, 1.195]
stroke = [1.222,  1.286, 2.250, 1.913, 1.895, 1.867, 1.827, 1.512, 2.283, 1.904, 1.713, 2.538, 1.885, 1.101, 2.336]

# Perform one-way ANOVA
f_stat, p_value = f_oneway(young, older, stroke)
print('F-statistic:', f_stat)
print('p-value:', p_value)

# Independent samples t-test
# Unequal sample sizes and assesses whether the means of the two groups are significantly different from each other.

# Young vs Old
t_statistic1, p_value1 = ttest_ind(young, older)
print("T-statistic:", t_statistic1)
print("P-value:", p_value1)

# Old vs Stroke
t_statistic2, p_value2 = ttest_ind(older, stroke)
print("T-statistic:", t_statistic2)
print("P-value:", p_value2)

# Young vs Stroke
t_statistic3, p_value3 = ttest_ind(young, stroke)
print("T-statistic:", t_statistic3)
print("P-value:", p_value3)

#############################
fig1, ax1 = plt.subplots()
ax1.grid(which='major', color=(0, 0, 0), linewidth=0.8, alpha=0.4)
ax1.grid(which='minor', color=(0, 0, 0), linestyle='--', linewidth=0.5, alpha=0.2)
ax1.minorticks_on()

# Plot a scatter plot for each row
ax1.scatter([1]*len(young), young, label='Young', alpha=0.6, s=100)
ax1.scatter([2]*len(older), older, label='Older', alpha=0.6, s=100)
ax1.scatter([3]*len(stroke), stroke, label='Stroke', alpha=0.6, s=100)

# Plot the error bar with mean and std
ax1.errorbar([1, 2, 3], [np.mean(young), np.mean(older), np.mean(stroke)],
             yerr=[np.std(young), np.std(older), np.std(stroke)],
             fmt='x', capsize=4, markersize = 15, elinewidth=2, capthick=0.5,
             label='Mean with Error Bars', color=(0, 0, 0))
xticks = [1, 2, 3]
xlabels = ['Young', 'Older', 'Stroke']
ax1.set_xticks(xticks, labels=xlabels, fontsize=16)
ax1.tick_params(axis='y', labelsize=16)

# Show figure
plt.ylabel('Pointing error (cm)', fontsize=16)


# Plot significance
# Adding significance lines and p-values
def add_pvalue(ax, x1, x2, y, pval, offset=0.1):
    """Draws lines with significance p-values on a Matplotlib plot."""
    line_x = np.array([x1, x1, x2, x2])
    line_y = np.array([y, y+offset, y+offset, y])
    ax.plot(line_x, line_y, lw=1.5, c='k')
    if pval < 0.05:
        if pval < 0.001:
            ax.text((x1 + x2)*.5, y + offset, "* p<0.001", ha='center', va='bottom', color='k', fontsize=16)
        else:
            ax.text((x1 + x2) * .5, y + offset, f"* p = {pval}", ha='center', va='bottom', color='k', fontsize=16)
    else:
        ax.text((x1 + x2) * .5, y + offset, f"p = {pval}", ha='center', va='bottom', color='k', fontsize=16)


ax = plt.gca()
ylim = ax.set_ylim([0.5, 3.3])

# Adjust the y positions based on your plot data
y_position = ylim[1] + (ylim[1] - ylim[0]) * 0.1

# Adding lines and annotations
add_pvalue(ax, 1, 2, 2.5, round(p_value1, 1))
add_pvalue(ax, 2, 3, 2.5 + 0.2, round(p_value2, 3))
add_pvalue(ax, 1, 3, 2.5 + 0.4, round(p_value3, 3))

plt.show()

