import json
from utils import get_rate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 设置支持负号显示

#'''     读取37个公司的股价数据
df = []
for i in range(1,38):
    tmp = pd.read_excel('附件1.xlsx',sheet_name='Sheet0 ({})'.format(i))
    df.append(tmp)
#'''




# 2021年的收盘价
'''1'''
spj = []
plt.figure(figsize=(3*3,7*4))
for i in range(21):
		spj.append(df[i]['收盘价'][-95:].values)
		plt.subplot(13, 3, i+1)
		plt.plot(spj[-1] ,label = df[i]['证券名称'].values[1],lw = 1,alpha = 0.9)
		plt.legend()
plt.show()

'''2'''
spj = []
plt.figure(figsize=(3*3,7*4))
for i in range(21,37):
		spj.append(df[i]['收盘价'][-95:].values)
		plt.subplot(13, 3, i-20)
		plt.plot(spj[-1] ,label = df[i]['证券名称'].values[1],lw = 1,alpha = 0.9)
		plt.legend()
plt.show()


'''总'''
spj = []
plt.figure(figsize=(9,13*3))
for i in range(37):
		spj.append(df[i]['收盘价'][-95:].values)
		plt.subplot(13, 3, i+1)
		plt.plot(spj[-1] ,label = df[i]['证券名称'].values[1],lw = 1,alpha = 0.9)
		plt.legend()
plt.show()


#%%

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



'''2021-01-20 —— 2021-05-27'''
date = {}
start_index = df[0][df[0]['交易时间'] == '2021-01-20'].index.values[0]
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
final_score_new = np.array(final_score)

corr=[]
for i in range(37):
		x = df[i]['涨跌幅%'].values[-83:]
		x = (x-x.mean())/x.std()
		y = (final_score_new-final_score_new.mean())/final_score_new.std()
		corr.append(pearsonr(x,y)[0]*3)
plt.figure(figsize=(10,3))
plt.plot(range(1,38),corr,'-o');plt.show()


np.random.randn(1)