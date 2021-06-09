import json
from utils import get_rate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 设置支持负号显示

#'''     读取37个公司的股价数据
df = []
for i in range(1,38):
    tmp = pd.read_excel('附件1.xlsx',sheet_name='Sheet0 ({})'.format(i))
    df.append(tmp)
#'''


#%% 前一模型，前段时间（2019-04-01 —— 2020-04-30）
with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)

'''各公司调整股本'''
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值

date = {}
start_index = df[0][df[0]['交易时间'] == '2019-04-01'].index.values[0]
end_index = df[0][df[0]['交易时间'] == '2020-04-30'].index.values[0]
for i in range(start_index,end_index+1):    #先获取日期
        ans = str(df[0]['交易时间'][i])[0:10]
        date[ans] = []


for dat in date.keys():
        each_day = []
        for i in range(37):
                try:
                        spj = df[i][df[i]['交易时间']==dat]['收盘价'].values[0]
                except:
                        spj = 0
                each_day.append(spj)
        date[dat] = each_day


base = 1000
tend = {}
#求基数
'''基数'''
base_num=0
for i,key in enumerate(fluent.keys()):
        base_num += date['2019-04-01'][i] * fluent[key]

for dat, lis in date.items():
        each_day_score = 0
        for i,coffe in enumerate(fluent.values()):
                each_day_score += lis[i]*coffe/base_num
        tend[dat] = each_day_score*base

'''2019-04-01 —— 2020-04-30 时间段内的板块指数'''
tend_array = np.array([i for i in tend.values()])


#%%后一模型，后段时间（2020-05-01 —— 2021-05-27）

#计算 new_div

# 2019-04-01 调整市值计算
with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2019-04-01）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2019-04-01']['收盘价'].values[0] * fluent[key]
'''总调整市值（2019-04-01）'''
total_fluent_3_ = sum([i for i in fluent.values()])



# 2021-05-05 调整市值计算
with open('./网上搜得的信息/各公司总股本（最新）.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本（最新）.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2021-05-06']['收盘价'].values[0] * fluent[key]
'''总调整市值（2021-05-06）'''
total_fluent_2_ = sum([i for i in fluent.values()])



#以 2019-04-01 的股本计算 2021-05-06
with open('./网上搜得的信息/各公司总股本.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
for i,key in enumerate(fluent.keys()):
        fluent[key] = df[i][df[i]['交易时间'] == '2021-05-06']['收盘价'].values[0] * fluent[key]
'''总调整市值（2019-05-06）'''
total_fluent_1_ = sum([i for i in fluent.values()])


'''计算新除数 new_div'''
new_div = total_fluent_3_*total_fluent_2_/total_fluent_1_
# 修正后的板块指数计算方法：  total_fluent_2_/new_div*1000



'''2020-05-06 —— 2021-05-27'''
date = {}
start_index = df[0][df[0]['交易时间'] == '2020-05-06'].index.values[0]
end_index = df[0][df[0]['交易时间'] == '2021-05-27'].index.values[0]
for i in range(start_index,end_index+1):    #先获取日期
        ans = str(df[0]['交易时间'][i])[0:10]
        date[ans] = []

#下面这段程序获取了问题1时间段内的收盘价，缺失的以0进行填充
for dat in date.keys():
        each_day = []
        for i in range(37):
                try:
                        spj = df[i][df[i]['交易时间']==dat]['收盘价'].values[0]
                except:
                        spj = 0
                each_day.append(spj)
        date[dat] = each_day
with open('./网上搜得的信息/各公司总股本（最新）.json') as f:
        dict_z = json.load(f)
with open('./网上搜得的信息/各公司流通股本（最新）.json') as f:
        dict_lt = json.load(f)
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值
'''求取每个公司的调整市值（2021-05-06）'''
final_score = []
fluent_value = {}


for dat in date.keys():
        each_day = []
        for i in range(37):
                try:
                        spj = df[i][df[i]['交易时间']==dat]['收盘价'].values[0]
                except:
                        spj = 0
                each_day.append(spj)
        date[dat] = each_day


mat = np.array([i for i in fluent.values()])
for dat in date.values():
				fluent_value = 0
				for i,key in enumerate(fluent.keys()):
								fluent_value = np.array(dat) * mat
				fluent_value = sum(fluent_value)
				final_score.append(fluent_value/new_div*1000)

'''板块指数new'''
final_score_new = final_score


#%% 读取上证指数(要注意这个时间的序号是反着的)
dataframe = pd.read_excel('上证指数.xlsx')
score = dataframe['收盘价'].values

# 前一时间段
tend_array = np.array(tend_array)
tend_array_real = np.array(score[-len(tend_array):].values)

# 后一时间段
final_score_new = np.array(final_score_new)
final_score_new_real = np.array(score[1:len(final_score_new)+1].values)


from scipy.stats import pearsonr
from minepy import MINE
from sklearn.preprocessing import MinMaxScaler



# 标准化
tend_array_std = (tend_array-tend_array.mean())/tend_array.std()
tend_array_real_std = (tend_array_real - tend_array_real.mean())/tend_array_real.std()

final_score_new_std = (final_score_new - final_score_new.mean())/final_score_new.std()
final_score_new_real_std = (final_score_new_real - final_score_new_real.mean())/final_score_new_real.std()


# 作图
def get_corr(tend_array_std,tend_array_real_std,final_score_new_std,final_score_new_real_std):
        '''
        plt.figure(figsize=(8,5))
        plt.plot(tend_array_std,label='光伏板块指数');plt.plot(tend_array_real_std,label = '上证指数');plt.legend()
        plt.title('标准化后 2019-04-01 至 2020-04-30 的上证指数与光伏板块指数走势对比');plt.show()
        plt.figure(figsize=(8,5))
        plt.plot(final_score_new_std,label='光伏板块指数');plt.plot(final_score_new_real_std,label = '上证指数');plt.legend()
        plt.title('标准化后 2020-05-06 至 2021-05-27 的上证指数与光伏板块指数走势对比');plt.show()
        '''

        #皮尔逊相关系数
        p1 = pearsonr(tend_array_std,tend_array_real_std)[0]
        p2 = pearsonr(final_score_new_std,final_score_new_real_std)[0]
        print('前段时间皮尔逊相关系数{:.4f}'.format(p1))
        print('后段时间皮尔逊相关系数{:.4f}'.format(p2))

        #互信息和最大信息系数

        m=MINE()
        m.compute_score(tend_array_std,tend_array_real_std)
        m1 = m.mic()
        print('前段时间最大信息系数{:.4f}'.format(m.mic()))
        m=MINE()
        m.compute_score(final_score_new_std,final_score_new_real_std)
        m2 = m.mic()
        print('后段时间最大信息系数{:.4f}'.format(m.mic()))

        #斯皮尔曼
        x1 = pd.Series(tend_array_std)
        y1 = pd.Series(tend_array_real_std)
        x2 = pd.Series(final_score_new_std)
        y2 = pd.Series(final_score_new_real_std)
        rsep1 = x1.corr(y1,method='spearman')
        rsep2 = x2.corr(y2,method='spearman')
        print('前段时间斯皮尔曼系数{:.4f}'.format(rsep1))
        print('后段时间斯皮尔曼系数{:.4f}'.format(rsep2))

        rkend1 = x1.corr(y1,method='kendall')
        rkend2 = x2.corr(y2,method='kendall')
        print('前段时间肯德尔系数{:.4f}'.format(rkend1))
        print('后段时间肯德尔系数{:.4f}'.format(rkend2))

        # 绝对值后平均
        avg1 = abs(p1)+abs(m1)+abs(rsep1)+abs(rkend1)
        avg2 = abs(p2)+abs(m2)+abs(rsep2)+abs(rkend2)
        print('前段时间取绝对值后平均值{:.4f}'.format(avg1/4))
        print('后段时间取绝对值后平均值{:.4f}'.format(avg2/4))
        print('\n')



# 数据
l1 = len(tend_array_std)
l2 = len(final_score_new)
for i in range(6):
        print('{} 月 - {} 月'.format(2*i,2*i+2))
        get_corr(tend_array_std[i:i+l1//6], tend_array_real_std[i:i+l1//6],
                 final_score_new_std[i:i+l2//6], final_score_new_real_std[i:i+l2//6])