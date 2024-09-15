import pandas as pd
import matplotlib.pyplot as plt

# 读取xlsx表格
df = pd.read_excel('./分组game/point之差.xlsx')

# 提取最后两列数据
values2 = df.iloc[:, -2].tolist()
values3 = df.iloc[:, -1].tolist()

# 绘制折线图
plt.figure(figsize=(8, 6))
plt.plot(df.index, values2, marker='o', color='b', label=df.columns[-2])
plt.plot(df.index, values3, marker='x', color='g', label=df.columns[-1])
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Last Two Columns Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
