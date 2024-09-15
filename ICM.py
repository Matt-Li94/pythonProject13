import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


dt_raw=pd.read_excel("./分组game/game1.xlsx")
diffa = dt_raw.loc[:,'pA_points_won']
diffb = dt_raw.loc[:,'pB_points_won']

dt_raw2 = pd.read_excel("./分组game/game6.xlsx")
diffa2 = dt_raw2.loc[:,'pA_points_won']
diffb2= dt_raw2.loc[:,'pB_points_won']

dt_raw3 = pd.read_excel("./分组game/game3.xlsx")
diffa3 = dt_raw3.loc[:,'pA_points_won']
diffb3 = dt_raw3.loc[:,'pB_points_won']

print(dt_raw)

x = dt_raw.loc[:,'elapsed_time']
x2 = dt_raw2.loc[:,'elapsed_time']
x3 = dt_raw3.loc[:,'elapsed_time']


diff=dt_raw.loc[:,'pA_points_won'].diff()

diff[0]=5

score1=diff+dt_raw.loc[:,'pA_ace']+dt_raw.loc[:,'pA_winner']+dt_raw.loc[:,'pA_net_pt_won']+dt_raw.loc[:,'pA_break_pt_won']
score1=score1+dt_raw.loc[:,'pA_double_fault']+dt_raw.loc[:,'pA_unf_err']+dt_raw.loc[:,'pA_break_pt_missed']
score1=pd.DataFrame({'score':score1})
score1['score']=score1['score'].cumsum(axis=0)

score1.insert(score1.shape[1],'time',pd.to_datetime(dt_raw.loc[:,'elapsed_time'],format='%H:%M:%S'))
diff2=dt_raw.loc[:,'pB_points_won'].diff()
diff2[0]=0
score2=diff2+dt_raw.loc[:,'pB_ace']+dt_raw.loc[:,'pB_winner']+dt_raw.loc[:,'pB_net_pt_won']+dt_raw.loc[:,'pB_break_pt_won']
score2=score2+dt_raw.loc[:,'pB_double_fault']+dt_raw.loc[:,'pB_unf_err']+dt_raw.loc[:,'pB_break_pt_missed']
score2=pd.DataFrame({'score':score2})
score2['score']=score2['score'].cumsum(axis=0)
score2.insert(score2.shape[1],'time',pd.to_datetime(dt_raw.loc[:,'elapsed_time'],format='%H:%M:%S'))

# 计算 pA_points_won 减去 pB_points_won 的结果
# score_diff = score1['score'] - score2['score']

score_diff = diffa - diffb
score_diff2 = diffa2 - diffb2
score_diff3 = diffa3 - diffb3


# # 将 score_diff 写入到文本文件
# with open('score1.txt', 'w') as file:
#     for item in score_diff:
#         file.write(str(item) + '\n')

# 绘制图表，并添加图例


# 绘制图表，并添加图例 -- 第一问
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(x)+1), score1['score'], label='Player 1')
plt.plot(range(1, len(x)+1), score2['score'], label='Player 2')
plt.plot(range(1, len(x)+1), score_diff, label='Score Difference (A - B)')
plt.title('Match1')
plt.xlabel('Time')
plt.ylabel('Score')
plt.legend()  # 添加图例
plt.show()
# plt.figure(figsize=(10, 6))
# plt.plot(x, score1['score'], label='Player 1')
# plt.plot(x, score2['score'], label='Player 2')
#
# plt.title('Match1')
# plt.legend()  # 添加图例
# plt.show()


# # 计算三者的相关系数
# correlation_matrix = pd.concat([score1['score'], score2['score'], score_diff], axis=1).corr()
#
# # 绘制热力图
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()
#
#
# # 添加行和列的标签
# plt.xticks([0.5, 1.5, 2.5], ['Player A Score', 'Player B Score', 'Score Difference (A - B)'])
# plt.yticks([0.5, 1.5, 2.5], ['Player A Score', 'Player B Score', 'Score Difference (A - B)'])
# plt.title('Correlation Heatmap')
#
# plt.show()
#
#
gap=score1.loc[:,'score']-score2.loc[:,'score']

acf_result = sm.tsa.stattools.acf(gap, nlags=5)

print(acf_result)
# # 绘制自相关函数图 --第二问
# plt.figure(figsize=(10, 6))
# plt.stem(range(len(acf_result)), acf_result, use_line_collection=True)
# plt.xlabel('Lag')
# plt.ylabel('ACF')
# plt.title('Autocorrelation Function (ACF)')
# plt.show()


# 设置子图
fig, ax = plt.subplots()

# 绘制 score_diff 曲线
ax.plot(range(1, len(x)+1), score_diff, label='Match 1',color='red',linewidth=0.8)



# 绘制 score_diff3 曲线
ax.plot(range(1, len(x3)+1), score_diff3, label='Match 3', color='green',linewidth=0.8)

# 绘制 score_diff2 曲线
ax.plot(range(1, len(x2)+1), score_diff2, label='Match 6',color='purple',linewidth=0.8)

# 添加标题和标签
ax.set_title('Score Difference Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Score Difference')

# 设置纵坐标范围
# ax.set_ylim(-20, 60)
ax.set_xlim(0, 300)

# 显示图例
ax.legend()

# 显示图形
plt.show()