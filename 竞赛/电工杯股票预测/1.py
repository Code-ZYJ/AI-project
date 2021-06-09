import pandas as pd
import json
import numpy as np
#%%
'''     读取37个公司的股价数据
df = []
for i in range(1,38):
    tmp = pd.read_excel('附件1.xlsx',sheet_name='Sheet0 ({})'.format(i))
    df.append(tmp)
'''

dict_z = {'南玻A' : 286327.72 ,
        '深圳能源' : 396449.16 ,
        '东旭蓝天' : 148687.39,
        '方大集团': 112338.42,
        '深赛格':  123565.62,
        '宝鹰股份': 134129.69,
        '东南网架': 103440.22,
        '延华智能' : 71215.30,
        '拓日新能' : 123634.21,
        '中利集团' : 87178.71,
        '亚厦股份' : 133999.65,
        '广田集团': 153727.97,
        '瑞和股份':36250.00,
        '亚玛顿': 16000.00,
        '永高股份': 112320.00,
        '中装建设':60000.00,
        '南网能源':303030.30,
        '特锐德': 99757.01,
        '嘉寓股份':71676.00,
        '东方日升':90430.19,
        '秀强股份':59295.24,
        '海达股份':60123.42,
        '旋极信息':171080.26,
        '中来股份':24099.47,
        '华自科技':26193.98,
        '启迪设计':13422.35,
        '汉嘉设计':21040.00,
        '精工钢构':181044.52,
        '苏美达':130674.94,
        '隆基股份':279079.52,
        '林洋能源':176541.78,
        '明阳智能':137972.24,
        '江河集团':115405,
        '中衡设计':27517.87,
        '森特股份':48001.2,
        '芯能科技':50000,
        '清源股份':27380,
}

dict_lt = {'南玻A' : 278564.87 ,
        '深圳能源' : 396449.16 ,
        '东旭蓝天' : 106034.06,
        '方大集团': 112195.26,
        '深赛格':  53802.49,
        '宝鹰股份': 126306.31,
        '东南网架': 95851.10,
        '延华智能' : 70996.81,
        '拓日新能' : 121571.73,
        '中利集团' : 70189.95,
        '亚厦股份' : 124272.89,
        '广田集团': 152929.58,
        '瑞和股份':30135.89,
        '亚玛顿': 15957.36,
        '永高股份':  89592.45,
        '中装建设':30606.00,
        '南网能源':75757.58,
        '特锐德': 92472.11,
        '嘉寓股份':71428.50,
        '东方日升':68254.06,
        '秀强股份':57471.52,
        '海达股份':45533.35,
        '旋极信息':105480.98,
        '中来股份':12057.77,
        '华自科技':19668.31,
        '启迪设计':11825.84,
        '汉嘉设计':5260,
        '精工钢构':151044.52,
        '苏美达':64028.40,
        '隆基股份':278173.69,
        '林洋能源':174887.68,
        '明阳智能':27590,
        '江河集团':115405,
        '中衡设计':26992.69,
        '森特股份':7501.2,
        '芯能科技':8800,
        '清源股份':8880,
}
with open('./网上搜得的信息/各公司总股本.json','w') as f:
        json.dump(dict_z,f,ensure_ascii=False)
with open('./网上搜得的信息/各公司流通股本.json','w') as f:
        json.dump(dict_lt,f,ensure_ascii=False)



# 定义分级靠档的函数求得加权比例
def get_rate(rate):
        rate = rate*100
        if rate<15:
                if rate%1>0:
                        rate = int(rate)+1
                else:
                        rate = int(rate)
        elif 15<rate<=20:
                rate = 20
        elif 20<rate<=30:
                rate = 30
        elif 30<rate<=40:
                rate = 40
        elif 40<rate<=50:
                rate = 50
        elif 50<rate<=60:
                rate = 60
        elif 60<rate<=70:
                rate = 70
        elif 70<rate<=80:
                rate = 80
        elif rate>80:
                rate = 100
        return rate/100
# 自由流通比例

'''各公司调整股本'''
fluent = {}
for key in dict_z.keys():
        fluent[key] = dict_z[key] * get_rate(dict_lt[key]/dict_z[key])    #每个股份的调整市值

# 饼图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

plt.figure(figsize=(10,10))
ans = [i for i in fluent.values()]
label = [i for i in fluent.keys()]
plt.pie(ans,labels=label,autopct='%1.1f%%',shadow=False,startangle=150)
plt.title('各证券流通股份占比')
plt.show()


#%% 板块指数
date = {}
start_index = df[0][df[0]['交易时间'] == '2019-04-01'].index.values[0]
end_index = df[0][df[0]['交易时间'] == '2021-04-30'].index.values[0]
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
with open('37家公司在2019-04-01__2021-04-30的收盘价.json','w') as f:  #以字典保存37个公司的收盘价
        json.dump(date,f,ensure_ascii=False)


#%%  假设基期指数为1000
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

tend_array = np.array([i for i in tend.values()])


def avg_line(days):
        arr = []
        for i in range(len(tend_array)-days+1):
                arr.append(sum(tend_array[i:i+days])/days)
        arr = np.array(arr)
        return arr
''' 5日移动平均线'''
d5 = avg_line(5)
'''10日移动平均线'''
d10 = avg_line(10)
'''20日移动平均线'''
d20 = avg_line(20)

#%% 作图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签


plt.plot(range(len(tend_array)),tend_array);plt.show();plt.xlim((0,len(tend_array)))
plt.plot(range(len(d5)),d5);plt.show()
plt.plot(range(len(d10)),d10);plt.show()
plt.plot(range(len(d20)),d20);plt.show()

#作图
plt.figure(figsize=(7,4),dpi = 150)
plt.plot(tend_array);plt.ylim((750,2100))
plt.title('光伏建筑一体化板块指数');plt.show()
plt.figure(figsize=(7,4),dpi = 150)
plt.plot(range(504),d5,'--',lw = 1.5,alpha = 0.8,label= ' 5日移动平均线')
plt.plot(range(5,504),d10,'--',lw = 1.5,alpha = 0.8,label= '10日移动平均线')
plt.plot(range(15,504),d20,'--',lw = 1.5,alpha = 0.8,label= '20日移动平均线')
plt.title('5日、10日、20日的移动平均线')
plt.ylim((750,2100));plt.legend();plt.show()