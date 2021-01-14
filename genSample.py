import pandas as pd
import matplotlib.pyplot as plt
import pandas_bokeh as pb
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, export_png, output_file
from bokeh.layouts import row, gridplot
from bokeh.models import ColumnDataSource
import numpy as np
import random

basedir = "/home/orin/Learning/Data"
dFilename = "2020年10月潜客数据.xlsx"
cFilename = "车型代码20201111_F.xlsx"
# databook = pd.read_excel("%s/%s" % (basedir, dFilename), sheet_name=0)
# databook.to_pickle("potential.pkl")
databook = pd.read_pickle("potential.pkl")

# 图表所需的层次
graphLVL = ["大区名称", "小区名称", "城市", "经销商代码"]

def summarySTAT():
    # 以后再写
    totalPotentials = databook['潜客量'].sum()
    # dealPotentialsbyDay = databook.groupby(['经销商代码', '日期'])['潜客量'].sum().reset_index()
    totalDealPotentials = databook.groupby(['经销商代码'])['潜客量'].sum().reset_index()
    totalPotentialsFIX = dataEsti['潜客量'].sum()
    # dealPotentialsbyDayFIX = dataEsti.groupby(['经销商代码', '日期'])['潜客量'].sum().reset_index()
    totalDealPotentialsFIX = dataEsti.groupby(['经销商代码'])['潜客量'].sum().reset_index()
    # dealDiffbyDay = dealPotentialsbyDay.merge(dealPotentialsbyDayFIX, on=['经销商代码', '日期'], suffixes=['_origin', '_fix'])
    dealDiff = totalDealPotentials.merge(totalDealPotentialsFIX, on=['经销商代码'], suffixes=['_origin', '_fix'])
    # dealDiffbyDay['差值'] = dealDiffbyDay['潜客量_origin'] - dealDiffbyDay['潜客量_fix']
    dealDiff['差值'] = dealDiff['潜客量_origin'] - dealDiff['潜客量_fix']
    print(dealDiff['差值'].value_counts())
    return totalPotentials - totalPotentialsFIX, dealDiff['差值'].value_counts()

def transArr(df):
    tmp = []
    for idx, row in df.iterrows():
        tmp.extend([row['车型new']] * row['潜客量'])
    return tmp

def genNewData(dCode, cSource, nowPOS):
    # dCode 经销商代码
    # cSource 客户来源更新
    # 进来后再做循环
    # 但是pandas dataframe插入行需要知道loc，所以return位置
    tmpArr = transArr(dataCOMPUTed[(dataCOMPUTed['经销商代码']==dCode)&(dataCOMPUTed['客户来源更新']==cSource)&(dataCOMPUTed['潜客量']>0)][['车型new', '潜客量']])
    # print(len(tmpArr))
    for idx, row in dataBYday[(dataBYday['经销商代码']==dCode)&(dataBYday['客户来源更新']==cSource)].iterrows():
        toDel = []
        # print(range(len(tmpArr)), row['潜客量'])
        for aIDX in random.sample(range(len(tmpArr)), int(row['潜客量'])):
            dataMiss.loc[nowPOS] = [
                pd.to_datetime(row['日期']),
                row['大区名称'],
                row['小区名称'],
                row['经销商代码'],
                row['经销商名称'],
                row['城市'],
                1,
                np.nan,
                row['客户来源更新'],
                tmpArr[aIDX]
            ]
            nowPOS +=1 
            toDel.append(aIDX)
        for i in sorted(toDel, reverse=True):
            del(tmpArr[i])
    return nowPOS

def tempDF(lvl, df):
    tmp = df[[lvl, '客户来源更新', '车型new', '潜客量']].copy()
    tmp = tmp.groupby([lvl, '客户来源更新', '车型new'])['潜客量'].sum().reset_index()
    tmp['%s汇总' % lvl] = tmp.groupby([lvl, '客户来源更新'])['潜客量'].transform('sum')
    tmp['车型占比'] = tmp['潜客量']/tmp['%s汇总' % lvl]
    tmp.drop(['潜客量', '%s汇总' % lvl], axis=1, inplace=True)
    return tmp

def batDraw(lvl):
    graphics = []
    vals = list(getIDX(lvl, dataUSEstd))
    df_o = tempDF(lvl, dataUSEstd)
    df_c = tempDF(lvl, dataEsti)

    for t in ["展厅", "线上"]:
        tmp = []
        for i in vals:
            tmp.append(drawGraph(
                df_o[(df_o[lvl]==i)&(df_o['客户来源更新']==t)][['车型new', '车型占比']],
                df_c[(df_c[lvl]==i)&(df_c['客户来源更新']==t)][['车型new', '车型占比']],
                '%s %s' % (i, t)
            ))
        graphics.append(tmp)

    return graphics

def drawGraph(se1, se2, title):
    # def drawGraph(tmpdf, title):
    tmpdf = se1.merge(se2, on=['车型new'], suffixes=['_origin', '_fix'])
    m = tmpdf.plot_bokeh(kind='barh', legend='top_right', x='车型new', title=title, show_figure=False, zooming=False)
    return m

def resetType(ser):
    if pd.isnull(ser['分组名称']):
        return np.nan
    else:
        return ser['车型']

def covTypes(df):
    for col in df.columns:
        if col == '潜客量':
            df[col] = df[col].astype('int32')
#         elif col== '日期':
#             df[col] = df[col].to_datetime()
        else:
            df[col] = df[col].astype('category')

def getIDX(cName, df, c=6):
    # 在df[cName]的最高30%里，随机取c个作为输出
    # 用于将来的分布验证
    tmp = df[[cName, '潜客量']].groupby([cName])['潜客量'].sum().reset_index()
    tmplist = df[cName].unique()
    if int(len(tmplist)*0.3) <= c:
        tmpidx = tmp.sort_values(by='潜客量', ascending=False).index.values
    else:
        tmpidx = tmp.sort_values(by='潜客量', ascending=False)[:int(len(tmplist)*0.3)].index.values
    return np.array(tmp[cName].iloc[random.sample(list(tmpidx), c)])

def balanceCom(rows):
    # 根据差值选择计算方式
    if rows['差值'] < 0:
        if rows['已进位'] and rows['顺序']+rows['差值']<=0:
            return rows['预计算潜客2'] - 1
    elif rows['差值'] > 0:
        if (not rows['已进位']) and rows['顺序'] - rows['差值'] <= 0:
            return rows['预计算潜客2'] + 1
    return rows['预计算潜客2']

def getColumns(col, vals,ty):
    # 获取画图的列
    oriData = dataUSEstd[(dataUSEstd[col]==vals)&(dataUSEstd['客户来源更新']==ty)][['车型new', '车型占比']]
    comData = dataEsti[(dataEsti[col]==vals)&(dataEsti['客户来源更新']==ty)][['车型new', '车型占比']]
    # oriData = statRES(col, dataUSEstd)[]
    return oriData.merge(comData, on='车型new', suffixes=['_origin', '_fix'])

# 将dataframe中的object类型转成 categories
covTypes(databook)
# 分组类型为空的部分设置成 NaN
databook['车型new'] = databook.apply(resetType, axis=1)
# dataAvailable 是原始数据中不需要处理的数据，主要是车型new有值的行
dataAvailable = databook[databook['车型new'].notna()]
dataAvailable.drop(['分组名称', '客户来源_原始', '车型', '月份'], axis=1, inplace=True)
# dataMiss 是需要生成的数据行，为了便于将来合并，需要将它的列设置为与
# dataAvailable 一致
dataMiss = pd.DataFrame(columns=list(dataAvailable.columns))

# 先按照经销商和客户来源进行数据推算，后按车型分布
dataUSE = databook[['客户来源更新', '大区名称', '小区名称', '城市', '经销商代码', '车型new', '潜客量']].copy()
cityLIST = dataUSE['城市'].unique()
area1LIST = dataUSE['大区名称'].unique()
area2LIST = dataUSE['小区名称'].unique()
dealerLIST = dataUSE['经销商代码'].unique()
# print(len(area1LIST), len(area2LIST), len(cityLIST), len(dealerLIST))

monthDays = len(databook['日期'].unique())
# dataBYday 是后续循环添加数据的list
dataBYday = databook[databook['车型new'].isna()].groupby(['日期', '大区名称', '小区名称', '经销商代码', '经销商名称', '城市', '客户来源更新'])['潜客量'].sum().reset_index()
# 计算 经销商 客户来源 和车型 的group by sum，去掉日期来计算概率
# sum(经销商、客户来源、车型)/monthDays 这就是某经销商在某来源下车型出现的概率
# dealerDISTbd.groupby(["经销商代码", "客户来源更新", "车型new", "日期"])["潜客量"].sum().reset_index()
# 按日的做起来太麻烦所以不用这个了
dataUSEstd = dataUSE[dataUSE['车型new'].notna()]
dataUSE2b = dataUSE[dataUSE['车型new'].isna()]
# 汇总车型数据
dataUSEstd = dataUSEstd.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新', '车型new'])['潜客量'].sum().reset_index()
# 创建一个基于车型的城市分布
# 因为后续会有部分经销商没有对应 "客户来源更新" 的分布，比如某些经销商没有线上的已有数据作为基准，这时就用城市的替换
# dataCITYstd = dataUSEstd.groupby(['城市', '客户来源更新', '车型new'])['潜客量'].sum().reset_index()
dataCITYstd = dataUSEstd.groupby(['城市', '车型new'])['潜客量'].sum().reset_index()
# 可能存在来源更新没有的情况，处理起来太复杂，直接用城市了
# dataCITYstd['城市汇总'] = dataCITYstd.groupby(['城市', '客户来源更新'])['潜客量'].transform('sum')
dataCITYstd['城市汇总'] = dataCITYstd.groupby(['城市'])['潜客量'].transform('sum')
dataCITYstd['车型占比'] = dataCITYstd['潜客量']/ dataCITYstd['城市汇总']
dataUSEstd['dealer汇总'] = dataUSEstd.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新'])['潜客量'].transform('sum')
dataUSEstd['车型占比'] = dataUSEstd['潜客量']/dataUSEstd['dealer汇总']
dataUSEstd['日均概率'] = dataUSEstd['潜客量']/monthDays

dataUSE2b = dataUSE2b.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新'])['潜客量'].sum().reset_index()

datatmp = dataUSE2b.copy()
datatmp = datatmp.merge(dataUSEstd[["经销商代码", "客户来源更新", "车型new", "车型占比"]], on=["经销商代码", "客户来源更新"], how='left')
# 获得车型占比后有些经销商没有对应的项，所以会变成NaN，需要使用前面已经算好的
# dataCITYstd 来进行合并（但其实这里也有可能存在NaN）
# 这样合并完后每个经销商就都有车型占比（分布）了
noneSTD = datatmp[datatmp['车型占比'].isna()]
datatmp = datatmp[datatmp['车型占比'].notna()]
noneSTD.drop(['车型new', '车型占比'], axis=1, inplace=True)
noneSTD = noneSTD.merge(dataCITYstd[['城市', '车型new', '车型占比']], on=['城市'])
datatmp = pd.concat([datatmp, noneSTD], ignore_index=True)
# 预计算潜客1 是未四舍五入的值， 预计算潜客2 是四舍五入后的值
datatmp['预计算潜客'] = datatmp["潜客量"] * datatmp["车型占比"]
datatmp['预计算潜客1'] = datatmp['预计算潜客'].map(int)
datatmp['预计算潜客2'] = datatmp['预计算潜客'].apply(lambda x: int(x+0.5))
datatmp['进位可能'] = datatmp['预计算潜客'] - datatmp['预计算潜客1']

##########################################
# 这里计算出来的 差值 就是四舍五入后的差异，需要用这个差异去修改 datatmp
# 里的 预计算潜客2 中不合适的部分。
# 并在 datatmp 中插入新的列 调整潜客
# 所以需要利用 进位可能，进行多退少补：
#     如果需要退n个（差值为负），则寻找 进位可能 > 0.5的数据里最小的n个各去掉 1
#     如果需要补n个（差值为正），则寻找 进位可能 < 0.5的数据里最大的n个各增加 1
# 这部分需要使用函数计算，感觉还比较麻烦
# 这里直接做一个 进位可能 相关的函数，主要是插入 顺序 字段，值是序数：
#   大于 0.5 的 升序，比如：
#     进位可能: [0.51, 0.66, 0.501, 0.7]
#        顺序: [1, 2, 0, 3]
#   小于 0.5 的 降序
##########################################
# 1. 给 datatemp 增加一列，区分 0.5
datatmp['已进位'] = datatmp['进位可能'] >= 0.5
# 都使用的升序，处理时要注意
# 先处理 已进位 为 True的
datatmp['顺序'] = 0
l = datatmp['顺序'] + datatmp[datatmp['已进位']==True].groupby(['经销商代码', '客户来源更新', '已进位'])['进位可能'].rank(ascending=True, method="first").astype('int')
r = datatmp['顺序'] + datatmp[datatmp['已进位']==False].groupby(['经销商代码', '客户来源更新', '已进位'])['进位可能'].rank(ascending=False, method="first").astype('int')
datatmp['顺序'] = l.fillna(0) + r.fillna(0)
# 计算经销商分来源的预计算潜客2的汇总
dataUSEpre = datatmp.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新'])['预计算潜客2'].sum().reset_index()
dataUSEadj = dataUSE2b.merge(dataUSEpre[['经销商代码', '客户来源更新', '预计算潜客2']], on=['经销商代码', '客户来源更新'])
dataUSEadj['差值'] = dataUSEadj['潜客量'] - dataUSEadj['预计算潜客2']
# 将 差值 合并进 datatmp
datatmp = datatmp.merge(dataUSEadj[['经销商代码', '客户来源更新', '差值']], on=['经销商代码', '客户来源更新'])
# 处理四舍五入的问题
datatmp['预计算真值'] = datatmp.apply(balanceCom, axis=1)
# 这里开始合并模拟结果和真实结果
dataCOMPUTed = datatmp[['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新','车型new', '预计算真值']].copy()
dataCOMPUTed.rename(columns={'预计算真值': '潜客量'}, inplace=True)

dataEsti = dataCOMPUTed.append(dataUSEstd[['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新','车型new', '潜客量']])
dataEsti = dataEsti.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新','车型new'])['潜客量'].sum().reset_index()
# 添加车型站比
dataEsti['dealer汇总'] = dataEsti.groupby(['大区名称', '小区名称', '城市', '经销商代码', '客户来源更新'])['潜客量'].transform('sum')
dataEsti['车型占比'] = dataEsti['潜客量']/dataEsti['dealer汇总']
dataEsti['日均概率'] = dataEsti['潜客量']/monthDays

dealerUnique = dataCOMPUTed[['大区名称', '小区名称', '经销商代码', '城市', '客户来源更新']].drop_duplicates()
dealerUnique = dealerUnique.reset_index(drop=True)[['经销商代码', '客户来源更新']]

# 生成数据
start = 0
for idx, row in dealerUnique.iterrows():
    print(idx, row['经销商代码'], row['客户来源更新'])
    start = genNewData(row['经销商代码'], row['客户来源更新'], start)
print(start)
# 这时再做合并就是类似dataAvailable一样了
newMiss = dataMiss.groupby(['日期', '大区名称', '小区名称', '经销商代码', '经销商名称', '城市', '客户来源更新', '车型new']).sum().reset_index()

# 画图，并生成html
pics = []
for pic in graphLVL:
    pics.extend(batDraw(pic))
html = pb.plot_grid(pics, plot_width=300, return_html=True)
with open("tt.html", 'w') as f:
    f.write(html)