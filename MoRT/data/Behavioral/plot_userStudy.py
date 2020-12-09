import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from matplotlib import rc
# from mort import dataMoral
import seaborn as sns

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
sns.set(style='ticks', palette='Set2')

#rc('text', usetex=True)

fontsize = 9
colors = sns.color_palette("Set2")

# fig = plt.figure(figsize=(4, 1.))
# ax = plt.gca()

# y_axis = [-3.9803878, -2.4381590, -0.8959301, 0.1836301, 1.4174132]
# minus_vars = [-4.285447, -2.628073, -0.987073, 0.104455, 1.270855]
# plus_vars = [-3.6753289, -2.2482450, -0.8047872, 0.2628052, 1.5639709]
# x_axis = [-5.0, -3.0, -1.0, 0.4, 2.0]

# ### diff_moral
# fig = plt.figure(figsize=(4, 3))
# ax = plt.gca()
# y_axis = [0.1499855, 0.0005962713, -0.1219029, -0.2683043, -0.3878157]
# minus_vars = [-0.05762404, -0.1397819, -0.2402308, -0.4203472, -0.5961913]
# plus_vars = [0.357595, 0.1409745, -0.003574944, -0.1162615, -0.1794402]
# x_axis = [-1.0, -0.5, -0.09, 0.40, 0.8]
# x_axis_labels = ["-1.0", "-0.5", "-0.09", "0.4", "0.8"]
#
# ax.fill_between(x_axis, minus_vars, plus_vars,
#                 color=colors[3],
#                 alpha=0.2)
#
# plt.plot(x_axis, y_axis, label="", color=colors[3], linewidth=2, linestyle="-")
#
#
# ax.set_xticks([-1.0, -0.5, 0.0, 0.5])
# ax.set_xticklabels(["-1.0", "-0.5", "0.0", "0.5"], fontsize=fontsize - 1)
# ax.set_yticks([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
# ax.set_yticklabels(["-0.6", "-0.4", "-0.2", "0.0", "0.2", "0.4"], fontsize=fontsize - 1)
# # ax.grid(zorder=0)
#
# ax.set_xlabel("diff_moral", fontsize=fontsize)
# ax.set_ylabel("rt_context_diff", fontsize=fontsize)
#
# plt.tight_layout()
# #plt.show()
# plt.savefig("./data/Behavioral/images/diff_moral.svg", dpi=600)

### answer
# fig = plt.figure(figsize=(4, 3))
# ax = plt.gca()
# y_axis = [-0.100669, -0.1358456]
# minus_vars = y_axis - np.array([-0.2272725, -0.2621776])
# plus_vars = np.array([0.02593442, -0.00951361]) - y_axis
# labels = ['No', 'Yes']
# x_axis = np.arange(len(labels))
#
# plt.errorbar(x_axis, y_axis, yerr=(minus_vars, plus_vars), label="", color=colors[1], linewidth=2, linestyle="-",
#              capsize=5, fmt='-o', visible=1)
#
# ax.set_xticks(x_axis)
# ax.set_xticklabels(labels, fontsize=fontsize - 1)
# ax.set_yticks([-0.25, -0.2, -0.15, -0.1, -0.05, 0])
# ax.set_yticklabels(["-0.25", "-0.2", "-0.15", "-0.1", "-0.05", "0"], fontsize=fontsize - 1)
# #ax.grid(zorder=0)
#
# ax.set_xlabel("answer", fontsize=fontsize)
# ax.set_ylabel("rt_context_diff", fontsize=fontsize)
#
# plt.tight_layout()
# #plt.show()
# plt.savefig("./data/Behavioral/images/answer.svg", dpi=600)

### answer_interaction
# fig = plt.figure(figsize=(4, 3))
# ax = plt.gca()
# y_axis = [-0.08970504, -0.1627614, 0.2627245, -0.1540456]
# minus_vars = y_axis - np.array([-0.2615001, -0.3314022, 0.001385058, -0.2894376])
# plus_vars = np.array([0.08209002, 0.005879417, 0.5240639, -0.01865364]) - y_axis
# labels = ['Neg - No', 'Pos - No', 'Neg - Yes', 'Pos - Yes']
# x_axis = np.arange(len(labels))
#
# plt.errorbar(x_axis, y_axis, yerr=(minus_vars, plus_vars), label="", color=colors[0], linewidth=2, linestyle="-",
#              capsize=5, fmt='-o', visible=1)
#
# ax.set_xticks(x_axis)
# ax.set_xticklabels(labels, fontsize=fontsize - 1)
# ax.set_yticks([-0.2, 0.0, 0.2, 0.4])
# ax.set_yticklabels(["-0.2", "0.0", "0.2", "0.4"], fontsize=fontsize - 1)
# #ax.grid(zorder=0)
#
# ax.set_xlabel("answer_interaction", fontsize=fontsize)
# ax.set_ylabel("rt_context_diff", fontsize=fontsize)
#
# plt.tight_layout()
# #plt.show()
# plt.savefig("./data/Behavioral/images/answer_interaction.svg", dpi=600)


### diff_moral
fig = plt.figure(figsize=(4, 3))
ax = plt.gca()
y_axis = [-3.536836, -2.164798, -0.7927592, 0.1676677, 1.265299]
minus_vars = [-4.043906, -2.481416, -0.9248429, 0.1080244, 1.086059]
plus_vars = [-3.029767, -1.848179, -0.6606756, 0.227311, 1.444538]
x_axis = [-5.0, -3.0, -1.0,  0.4,  2.0]
x_axis_labels = ["-5.0", "-3.0", "-1.0", "0.4", "2.0"]

ax.fill_between(x_axis, minus_vars, plus_vars,
                color=colors[4],
                alpha=0.2)

plt.plot(x_axis, y_axis, label="", color=colors[4], linewidth=2, linestyle="-")


ax.set_xticks([-5, -4, -3, -2, -1, 0, 1, 2])
ax.set_xticklabels(["-5", "-4", "-3", "-2", "-1", "0", "1", "2"], fontsize=fontsize - 1)
ax.set_yticks([-4, -3, -2, -1, 0, 1])
ax.set_yticklabels(["-4", "-3", "-2", "-1", "0", "1"], fontsize=fontsize - 1)
# ax.grid(zorder=0)

ax.set_xlabel("mean_rt_context_diff", fontsize=fontsize)
ax.set_ylabel("rt_atomic_diff", fontsize=fontsize)

plt.tight_layout()
#plt.show()
plt.savefig("./data/Behavioral/images/mean_rt_context_diff.svg", dpi=600)





# ax.set_xticks(x_axis)
# ax.set_xticklabels(labels, fontsize=fontsize - 1)
# ax.set_yticks([-0.2, -0.15, -0.1, -0.05, 0])
# ax.set_yticklabels(["-0.2", "-0.15", "-0.1", "-0.05", "0"], fontsize=fontsize - 1)
# #ax.set_xticks([10, 100, 1000, 10000])
# #ax.set_xticklabels(["10", "100", "1K", "10K"], fontsize=fontsize - 1)
# #ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# #ax.set_yticklabels(["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], fontsize=fontsize - 1)
#
# ### diff_moral
# ax.set_xlabel("diff_moral", fontsize=fontsize)
# ax.set_ylabel("rt_context_diff", fontsize=fontsize)
#
# plt.tight_layout()
# plt.show()
#plt.savefig("./data/Behavioral/images/diff_moral.svg", dpi=600)
