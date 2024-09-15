import matplotlib.pyplot as plt
import numpy as np

matches = ['1', '2', '3', '4', '5', '6']
values1 = [1, 0.953913, 0.91073739, 0.87240726, 0.83886623, 0.8051506]
values2 = [1, 0.98554961, 0.97246763, 0.95941921, 0.94567994, 0.93211566]
values3 = [1, 0.96707877, 0.9332760, 0.89916693, 0.86563572, 0.83499515]

colors = ['lightcoral', 'lightgreen', 'lightblue', 'lightcyan', 'lightpink', 'yellow']

barWidth = 0.4
y = np.arange(len(values1))

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

axs[0].barh(y, values1, color=colors, height=barWidth)
axs[0].set_yticks(y)
axs[0].set_yticklabels(matches)
axs[0].set_xlabel('acf')
axs[0].set_ylabel('Match1')
axs[0].set_title('acf for Match 1')
# 添加数值标签
for i, v in enumerate(values1):
    axs[0].text(v, i, str(round(v, 3)), color='black', va='center')

axs[1].barh(y, values2, color=colors, height=barWidth)
axs[1].set_yticks(y)
axs[1].set_yticklabels(matches)
axs[1].set_xlabel('acf')
axs[1].set_ylabel('Match3')
axs[1].set_title('acf for Match 3')
# 添加数值标签
for i, v in enumerate(values2):
    axs[1].text(v, i, str(round(v, 3)), color='black', va='center')

axs[2].barh(y, values3, color=colors, height=barWidth)
axs[2].set_yticks(y)
axs[2].set_yticklabels(matches)
axs[2].set_xlabel('acf')
axs[2].set_ylabel('Match6')
axs[2].set_title('acf for Match 6')
# 添加数值标签
for i, v in enumerate(values3):
    axs[2].text(v, i, str(round(v, 3)), color='black', va='center')

plt.tight_layout()
plt.show()





# 只用一种颜色
# import matplotlib.pyplot as plt
# import numpy as np
#
# matches = ['1', '2', '3', '4', '5', '6']
# values1 = [1, 0.953913, 0.91073739, 0.87240726, 0.83886623, 0.8051506]
# values2 = [1, 0.98554961, 0.97246763, 0.95941921, 0.94567994, 0.93211566]
# values3 = [1, 0.96707877, 0.9332760, 0.89916693, 0.86563572, 0.83499515]
#
# barWidth = 0.4
# y = np.arange(len(values1))
#
# fig, axs = plt.subplots(3, 1, figsize=(8, 6))
#
# def calculate_color(value):
#     green = (0, 1, 0)  # 绿色
#     intensity = 1 - value  # 颜色深浅与数值大小相关
#     return (green[0], green[1] * intensity, green[2] * intensity)
#
# for i in range(len(matches)):
#     axs[0].barh(i, values1[i], color=calculate_color(values1[i]), height=barWidth)
#     axs[1].barh(i, values2[i], color=calculate_color(values2[i]), height=barWidth)
#     axs[2].barh(i, values3[i], color=calculate_color(values3[i]), height=barWidth)
#
# axs[0].set_yticks(y)
# axs[0].set_yticklabels(matches)
# axs[0].set_xlabel('Scores')
# axs[0].set_ylabel('Match1')
# axs[0].set_title('Scores for Match 1')
#
# axs[1].set_yticks(y)
# axs[1].set_yticklabels(matches)
# axs[1].set_xlabel('Scores')
# axs[1].set_ylabel('Match3')
# axs[1].set_title('Scores for Match 3')
#
# axs[2].set_yticks(y)
# axs[2].set_yticklabels(matches)
# axs[2].set_xlabel('Scores')
# axs[2].set_ylabel('Match6')
# axs[2].set_title('Scores for Match 6')
#
# plt.tight_layout()
# plt.show()
