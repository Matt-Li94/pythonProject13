import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdate
import torch
from sklearn.model_selection import train_test_split
from torch import nn

# 处理数据
dt_raw = pd.read_excel("./分组game/game1.xlsx")
diff = dt_raw.loc[:, 'pA_points_won'].diff()

score1 = diff + dt_raw.loc[:, 'pA_ace'] + dt_raw.loc[:, 'pA_winner'] + dt_raw.loc[:, 'pA_net_pt_won'] + \
         dt_raw.loc[:, 'pA_break_pt_won']
score1 = score1 + dt_raw.loc[:, 'pA_double_fault'] + dt_raw.loc[:, 'pA_unf_err'] + \
         dt_raw.loc[:, 'pA_break_pt_missed']
score1 = pd.DataFrame({'score': score1})
score1['score'] = score1['score'].cumsum(axis=0)
score1.insert(score1.shape[1], 'time', pd.to_datetime(dt_raw.loc[:, 'elapsed_time'], format='%H:%M:%S'))
p1_train = pd.DataFrame({'ace': dt_raw.loc[:, 'pA_ace'], 'winner': dt_raw.loc[:, 'pA_winner'],
                         'net': dt_raw.loc[:, 'pA_net_pt_won'], 'break': dt_raw.loc[:, 'pA_break_pt_won'],
                         'unf': dt_raw.loc[:, 'pA_unf_err'],'double': dt_raw.loc[:, 'pA_double_fault'],
                         'miss': dt_raw.loc[:, 'pA_break_pt_missed'],
                         'point': dt_raw.loc[:, 'pA_points_won']})

diff2 = dt_raw.loc[:, 'pB_points_won'].diff()
diff2[0] = 0
score2 = diff + dt_raw.loc[:, 'pB_ace'] + dt_raw.loc[:, 'pB_winner'] + dt_raw.loc[:, 'pB_net_pt_won'] + \
         dt_raw.loc[:, 'pB_break_pt_won']
score2 = score2 + dt_raw.loc[:, 'pB_double_fault'] + dt_raw.loc[:, 'pB_unf_err'] + \
         dt_raw.loc[:, 'pB_break_pt_missed']
score2 = pd.DataFrame({'score': score2})
score2['score'] = score2['score'].cumsum(axis=0)
score2.insert(score2.shape[1], 'time', pd.to_datetime(dt_raw.loc[:, 'elapsed_time'], format='%H:%M:%S'))
p2_train = pd.DataFrame({'ace': dt_raw.loc[:, 'pB_ace'], 'winner': dt_raw.loc[:, 'pB_winner'],
                         'net': dt_raw.loc[:, 'pB_net_pt_won'], 'break': dt_raw.loc[:, 'pB_break_pt_won'],
                          'unf': dt_raw.loc[:, 'pB_unf_err'],'double': dt_raw.loc[:, 'pB_double_fault'],
                         'miss': dt_raw.loc[:, 'pB_break_pt_missed'],
                         'point': dt_raw.loc[:, 'pB_points_won']})

gap = score1.loc[:, 'score'] - score2.loc[:, 'score']
gap = pd.DataFrame({'gap': gap})
gap_points = dt_raw.loc[:, 'pA_points_won'] - dt_raw.loc[:, 'pB_points_won']
gap_points = pd.DataFrame({'gap': gap_points})
train = p1_train - p2_train

x_train, x_test, y_train, y_test = train_test_split(train, gap_points, test_size=0.2, random_state=42)
# 转换数据类型为torch.Tensor
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# LSTM
class Lstm(nn.Module):
    def __init__(self, input, hidden, output):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input, hidden)
        self.linear = nn.Linear(hidden, output)
        self.act = nn.Tanh()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.act(out)
        out = self.linear(out)
        return out


model2 = Lstm(8, 32, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)

num_epochs = 400 ## 训练轮数 (拟合效果主要改动这个)
for epoch in range(num_epochs):
    outputs = model2(x_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % (num_epochs / 10) == 0:
        print(loss)

# 使用模型进行预测
train = torch.tensor(train.values, dtype=torch.float32)
model2.eval()
pre = model2(train)
pre = pre.detach().numpy()
# print(pre.squeeze(1))
plant=pd.DataFrame({'predicted':pre.squeeze(1),'real':gap_points.loc[:,'gap']})
plant.insert(plant.shape[1], 'time', pd.to_datetime(dt_raw.loc[:, 'elapsed_time'], format='%H:%M:%S'))
#plt.style.use('whitegrid')
sns.set(style='whitegrid')

fig1 = plt.figure(figsize=(15,5))
ax = fig1.add_subplot(1,1,1)
ax.xaxis.set_major_formatter(mdate.DateFormatter('%H:%M'))

sns.lineplot(x='time',y='real',label='real',linestyle='solid',data=plant,linewidth=5,color='crimson') # 线宽改linewidth
sns.lineplot(x='time',y='predicted',label='predicted',linestyle='dotted',data=plant,linewidth=7,color='darkorchid')
plt.xlabel('Time')
plt.ylabel('Advantage')
name=['0:00','0:30','1:00','1:30','2:00','2:30']
plt.legend(loc='upper left')
plt.show()

