# -*-coding:utf-8-*-
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

# DRAGONS
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

import ast
from tqdm import tqdm
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

import os

random_seed = 2019


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


## load data
train = pd.read_csv('../data/prediction/train.csv')
test = pd.read_csv('../data/prediction/test.csv')
# train.index = train['id']
# test.index = test['id']

print("Dimension of train : " + str(train.shape) + " || Dimension of test : " + str(test.shape))
# print(train.head())

train.loc[train['id'] == 16, 'revenue'] = 192864
train.loc[train['id'] == 90, 'budget'] = 30000000
train.loc[train['id'] == 118, 'budget'] = 60000000
train.loc[train['id'] == 149, 'budget'] = 18000000
train.loc[train['id'] == 313, 'revenue'] = 12000000
train.loc[train['id'] == 451, 'revenue'] = 12000000
train.loc[train['id'] == 464, 'budget'] = 20000000
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 930000
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 819, 'budget'] = 90000000
train.loc[train['id'] == 850, 'budget'] = 90000000
train.loc[train['id'] == 1007, 'budget'] = 2
train.loc[train['id'] == 1112, 'budget'] = 7500000
train.loc[train['id'] == 1131, 'budget'] = 4300000
train.loc[train['id'] == 1359, 'budget'] = 10000000
train.loc[train['id'] == 1542, 'budget'] = 1
train.loc[train['id'] == 1570, 'budget'] = 15800000
train.loc[train['id'] == 1571, 'budget'] = 4000000
train.loc[train['id'] == 1714, 'budget'] = 46000000
train.loc[train['id'] == 1721, 'budget'] = 17500000
train.loc[train['id'] == 1865, 'revenue'] = 25000000
train.loc[train['id'] == 1885, 'budget'] = 12
train.loc[train['id'] == 2091, 'budget'] = 10
train.loc[train['id'] == 2268, 'budget'] = 17500000
train.loc[train['id'] == 2491, 'budget'] = 6
train.loc[train['id'] == 2602, 'budget'] = 31000000
train.loc[train['id'] == 2612, 'budget'] = 15000000
train.loc[train['id'] == 2696, 'budget'] = 10000000
train.loc[train['id'] == 2801, 'budget'] = 10000000
train.loc[train['id'] == 335, 'budget'] = 2
train.loc[train['id'] == 348, 'budget'] = 12
train.loc[train['id'] == 470, 'budget'] = 13000000
train.loc[train['id'] == 513, 'budget'] = 1100000
train.loc[train['id'] == 640, 'budget'] = 6
train.loc[train['id'] == 696, 'budget'] = 1
train.loc[train['id'] == 797, 'budget'] = 8000000
train.loc[train['id'] == 850, 'budget'] = 1500000
train.loc[train['id'] == 1199, 'budget'] = 5
train.loc[train['id'] == 1282, 'budget'] = 9
train.loc[train['id'] == 1347, 'budget'] = 1
train.loc[train['id'] == 1755, 'budget'] = 2
train.loc[train['id'] == 1801, 'budget'] = 5
train.loc[train['id'] == 1918, 'budget'] = 592
train.loc[train['id'] == 2033, 'budget'] = 4
train.loc[train['id'] == 2118, 'budget'] = 344
train.loc[train['id'] == 2252, 'budget'] = 130
train.loc[train['id'] == 2256, 'budget'] = 1
train.loc[train['id'] == 2696, 'budget'] = 10000000

test.loc[test['id'] == 3033, 'budget'] = 250
test.loc[test['id'] == 3051, 'budget'] = 50
test.loc[test['id'] == 3084, 'budget'] = 337
test.loc[test['id'] == 3224, 'budget'] = 4
test.loc[test['id'] == 3594, 'budget'] = 25
test.loc[test['id'] == 3619, 'budget'] = 500
test.loc[test['id'] == 3831, 'budget'] = 3
test.loc[test['id'] == 3935, 'budget'] = 500
test.loc[test['id'] == 4049, 'budget'] = 995946
test.loc[test['id'] == 4424, 'budget'] = 3
test.loc[test['id'] == 4460, 'budget'] = 8
test.loc[test['id'] == 4555, 'budget'] = 1200000
test.loc[test['id'] == 4624, 'budget'] = 30
test.loc[test['id'] == 4645, 'budget'] = 500
test.loc[test['id'] == 4709, 'budget'] = 450
test.loc[test['id'] == 4839, 'budget'] = 7
test.loc[test['id'] == 3125, 'budget'] = 25
test.loc[test['id'] == 3142, 'budget'] = 1
test.loc[test['id'] == 3201, 'budget'] = 450
test.loc[test['id'] == 3222, 'budget'] = 6
test.loc[test['id'] == 3545, 'budget'] = 38
test.loc[test['id'] == 3670, 'budget'] = 18
test.loc[test['id'] == 3792, 'budget'] = 19
test.loc[test['id'] == 3881, 'budget'] = 7
test.loc[test['id'] == 3969, 'budget'] = 400
test.loc[test['id'] == 4196, 'budget'] = 6
test.loc[test['id'] == 4221, 'budget'] = 11
test.loc[test['id'] == 4222, 'budget'] = 500
test.loc[test['id'] == 4285, 'budget'] = 11
test.loc[test['id'] == 4319, 'budget'] = 1
test.loc[test['id'] == 4639, 'budget'] = 10
test.loc[test['id'] == 4719, 'budget'] = 45
test.loc[test['id'] == 4822, 'budget'] = 22
test.loc[test['id'] == 4829, 'budget'] = 20
test.loc[test['id'] == 4969, 'budget'] = 20
test.loc[test['id'] == 5021, 'budget'] = 40
test.loc[test['id'] == 5035, 'budget'] = 1
test.loc[test['id'] == 5063, 'budget'] = 14
test.loc[test['id'] == 5119, 'budget'] = 2
test.loc[test['id'] == 5214, 'budget'] = 30
test.loc[test['id'] == 5221, 'budget'] = 50
test.loc[test['id'] == 4903, 'budget'] = 15
test.loc[test['id'] == 4983, 'budget'] = 3
test.loc[test['id'] == 5102, 'budget'] = 28
test.loc[test['id'] == 5217, 'budget'] = 75
test.loc[test['id'] == 5224, 'budget'] = 3
test.loc[test['id'] == 5469, 'budget'] = 20
test.loc[test['id'] == 5840, 'budget'] = 1
test.loc[test['id'] == 5960, 'budget'] = 30
test.loc[test['id'] == 6506, 'budget'] = 11
test.loc[test['id'] == 6553, 'budget'] = 280
test.loc[test['id'] == 6561, 'budget'] = 7
test.loc[test['id'] == 6582, 'budget'] = 218
test.loc[test['id'] == 6638, 'budget'] = 5
test.loc[test['id'] == 6749, 'budget'] = 8
test.loc[test['id'] == 6759, 'budget'] = 50
test.loc[test['id'] == 6856, 'budget'] = 10
test.loc[test['id'] == 6858, 'budget'] = 100
test.loc[test['id'] == 6876, 'budget'] = 250
test.loc[test['id'] == 6972, 'budget'] = 1
test.loc[test['id'] == 7079, 'budget'] = 8000000
test.loc[test['id'] == 7150, 'budget'] = 118
test.loc[test['id'] == 6506, 'budget'] = 118
test.loc[test['id'] == 7225, 'budget'] = 6
test.loc[test['id'] == 7231, 'budget'] = 85
test.loc[test['id'] == 5222, 'budget'] = 5
test.loc[test['id'] == 5322, 'budget'] = 90
test.loc[test['id'] == 5350, 'budget'] = 70
test.loc[test['id'] == 5378, 'budget'] = 10
test.loc[test['id'] == 5545, 'budget'] = 80
test.loc[test['id'] == 5810, 'budget'] = 8
test.loc[test['id'] == 5926, 'budget'] = 300
test.loc[test['id'] == 5927, 'budget'] = 4
test.loc[test['id'] == 5986, 'budget'] = 1
test.loc[test['id'] == 6053, 'budget'] = 20
test.loc[test['id'] == 6104, 'budget'] = 1
test.loc[test['id'] == 6130, 'budget'] = 30
test.loc[test['id'] == 6301, 'budget'] = 150
test.loc[test['id'] == 6276, 'budget'] = 100
test.loc[test['id'] == 6473, 'budget'] = 100
test.loc[test['id'] == 6842, 'budget'] = 30

# external data
release_dates = pd.read_csv('../data/prediction/release_dates_per_country.csv')
release_dates['id'] = range(1, 7399)
release_dates.drop(['original_title', 'title'], axis=1, inplace=True)
# release_dates.index = release_dates['id']
print(release_dates.head())
print(train.head())
train = pd.merge(train, release_dates, how='left', on=['id'])
test = pd.merge(test, release_dates, how='left', on=['id'])

trainAdditionalFeatures = pd.read_csv('../data/prediction/TrainAdditionalFeatures.csv')[
    ['imdb_id', 'popularity2', 'rating']]
testAdditionalFeatures = pd.read_csv('../data/prediction/TestAdditionalFeatures.csv')[
    ['imdb_id', 'popularity2', 'rating']]

train = pd.merge(train, trainAdditionalFeatures, how='left', on=['imdb_id'])
test = pd.merge(test, testAdditionalFeatures, how='left', on=['imdb_id'])

print(train.head())

'''
### 计算了训练集中的相关系数

可以发现：
    1、票房与投资的相关系数最高，达到了0.755986
    2、票房与theatrical的相关系数很高，达到了0.552502
    3、票房与popularity2的相关系数很高，达到了0.629664
相比之下一些无关紧要的因素如id,release_year等则很低，说明数据比较可靠。
'''
print(train.corr())

'''
### 票房与投资的关系

从图中可以看出，票房与投资具有很强的相关关系
'''

x1 = np.array(train["budget"])
y1 = np.array(train["revenue"])

fig = plt.figure(1, figsize=(9, 5))

# plt.plot([0,400000000],[0,400000000],c="green")
plt.scatter(x1, y1, c=['green'], marker='o')
plt.grid()
plt.xlabel("budget", fontsize=10)
plt.ylabel("revenue", fontsize=10)
plt.title("Link between revenue and budget", fontsize=10)
plt.savefig('./figures/revenue_budget.png')

### 票房与popularity2的关系
x1 = np.array(train["popularity2"])
y1 = np.array(train["revenue"])

fig = plt.figure(1, figsize=(9, 5))

# plt.plot([0,400000000],[0,400000000],c="green")
plt.scatter(x1, y1, c=['green'], marker='o')
plt.grid()
plt.xlabel("popularity", fontsize=10)
plt.ylabel("revenue", fontsize=10)
plt.title("Link between popularity and revenue", fontsize=10)
plt.savefig('./figures/revenue_popularity.png')


# 票房与theatrical的关系
x1 = np.array(train["theatrical"])
y1 = np.array(train["revenue"])

fig = plt.figure(1, figsize=(9, 5))

# plt.plot([0,400000000],[0,400000000],c="green")
plt.scatter(x1, y1, c=['green'], marker='o')
plt.grid()
plt.xlabel("theatrical", fontsize=10)
plt.ylabel("revenue", fontsize=10)
plt.title("Link between theatrical and revenue", fontsize=10)
plt.savefig('./figures/revenue_theatrical.png')


# 票房与语种的关系
plt.figure(figsize=(15,11)) #figure size

#It's another way to plot our data. using a variable that contains the plot parameters
g1 = sns.boxenplot(x='original_language', y='revenue',
                   data=train[(train['original_language'].isin((train['original_language'].value_counts()[:10].index.values)))])
g1.set_title("Revenue by original language's movies", fontsize=20) # title and fontsize
g1.set_xticklabels(g1.get_xticklabels(),rotation=45) # It's the way to rotate the xticks when we use variable to our graphs
g1.set_xlabel('Original language', fontsize=18) # Xlabel
g1.set_ylabel('Revenue', fontsize=18) #Ylabel
plt.savefig('./figures/revenue_language.png')

# 投资与年份

# print(train[(train['release_year'].isin(train['release_year'].value_counts()[:67].index.values))].head())
# print(train['release_year'].value_counts().index)

(sns.FacetGrid(train[(train['release_year']\
                        .isin(train['release_year']\
                              .value_counts()[:5].index.values))],
               hue='release_year', height=5, aspect=2)
  .map(sns.kdeplot, 'budget', shade=True)
 .add_legend()
)
plt.title("Budget by all years")
plt.savefig('./figures/budget_recent_year.png')


# 票房与年份

plt.figure(figsize=(12,5))

# Subplot allow us to plot more than one
# in this case, will be create a subplot grid of 2 x 1

# seting the distribuition of our data and normalizing using np.log on values highest than 0 and +
# also, we will set the number of bins and if we want or not kde on our histogram
ax = sns.distplot(np.log1p(train['revenue']), bins=40, kde=True)
ax.set_xlabel('Revenue', fontsize=15) #seting the xlabel and size of font
ax.set_ylabel('Distribuition', fontsize=15) #seting the ylabel and size of font
ax.set_title("Distribuition of Revenue", fontsize=20) #seting the title and size of font
plt.savefig('./figures/revenue_year.png')



# 相关关系可视化
col = ['revenue','budget','popularity2','theatrical','runtime','id', 'release_year']
plt.subplots(figsize=(14, 10))
corr = train[col].corr()

sns.heatmap(corr, xticklabels=col, yticklabels=col, linewidths=.5, cmap="Reds")
plt.savefig('./figures/corre.png')