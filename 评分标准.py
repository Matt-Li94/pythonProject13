#绘制寻优图的代码

import matplotlib.pyplot as plt

data = [1, 2, 2, 3, -5, -5, -8]
indices = list(range(len(data)))
labels = ['Buwse', 'Buwsh', 'Bwnet', 'Bwse', 'Dmb', 'Due', 'Dmo']  # 设置横坐标的标签

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'orange']  # 定义颜色列表

plt.bar(indices, data, color=colors)  # 使用颜色列表
plt.ylabel('Value')

# 在每个柱状图上标注数值
for i, v in enumerate(data):
    if v < 0:
        plt.text(i, v, str(v), ha='center', va='top')  # 将负值放在柱形的上方
    else:
        plt.text(i, v, str(v), ha='center', va='bottom')  # 将非负值放在柱形的下方

# 设置横坐标的标签
plt.xticks(indices, labels)

# 添加一条黑色横向参考线
plt.axhline(color='black', linewidth=1.0)

plt.show()




