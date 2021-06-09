import pandas as pd
import matplotlib.pyplot as plt
import ffn  #金融计算包
import tushare as ts#获取金融数据的工具包
plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

#%% 获取股票数据
chinaBank = ts.get_hist_data('601988', '2018-01-01', '2019-01-01')  #中国银行
chinaBank = chinaBank.sort_values(by='date', ascending=True)  #数据转化为升序
Close = chinaBank.close
Close.head()

#%% 收益率
#3.1 一期收益率
#将索引值变换成日期型数据(datetime)，
Close.index = pd.to_datetime(Close.index)
#收盘价格滞后一期，第一位数据由于没有前项，值会变为NaN
lagClose = Close.shift(1)
#将收盘价格与滞后一期的收盘价格合并，转换成DataFrame数据
Close_hebing = pd.DataFrame({"Close": Close, "lagClose": lagClose})
Close_hebing.head()
#收益率
simpleret = (Close - lagClose) / lagClose
simpleret.name = 'simpleret'
#中国银行一期收益率
simpleret.head()  #每天的收益率

#3.2 二期收益率
simpleret2 = (Close - Close.shift(2)) / Close.shift(2)
simpleret2.name = 'simpleret2'
simpleret2.head()

#3.3 单期收益率曲线图
plt.figure(figsize=(10, 6))
simpleret.plot()

#3.4 累积（多期）收益率曲线图
plt.figure(figsize=(10, 6))
((1 + simpleret).cumprod() - 1).plot()  #累乘cumprod并绘图

#%% 年化收益
#累加cumsum和累乘cumprod
#年华收益率计算公式：[（1+r1）*(1+r2)*...(1+rn)]**(n/m)，n为一年股票交易天数，m为大盘交易天数
annualize = (1 + simpleret).cumprod()[-1]**(245 / 311) - 1
print("中国银行2018年年收益:" + str(annualize))

#%% 风险度量
# 5.1 度量方式1——方差度量风险
#方差度量风险，相当于是数据的稳定性，这里转化为收益的稳定性。
returnS = ffn.to_returns(chinaBank.close).dropna() #计算一期收益率
print("中国银行方差风险：" + str(returnS.std()**2)) #std()函数是标准差，需要平方

# 5.2 度量方式1——下行风险
#这里自定义了下行风险偏差函数。无风险收益率不仅可以用自身的平均收益率，还可以使用各个典型的银行定期收益率作为无风险收益率。
#下行偏差风险函数，返回值越大则对应的风险越大
def cal_down_risk(returns):
    mu = returns.mean()#无风险利率，这里取平均值
    temp = returns[returns < mu]
    down_risk = (sum((mu - temp)**2) / len(returns))**0.5
    return (down_risk)
print("下行风险：" + str(cal_down_risk(returnS)))