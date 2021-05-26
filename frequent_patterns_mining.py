#!/usr/bin/env python
# coding: utf-8

# # 数据挖掘互评作业二: 频繁模式与关联规则挖掘

# ## 一、读取数据集，并查看数据集的信息概要

# In[1]:


import pandas as pds
data = pds.read_csv('./Wine Reviews/winemag-data-130k-v2.csv')
data.info()


# ## 二、根据数据集的信息筛选出进行频繁模式挖掘的列‘
# 
# 筛选依据为：
# * id 列不具有实际意义
# * description、title列完全不具有重复的数值
# * price 列为浮点数，region_1 和 winery 列数值重复度低
# * desgination、region_2和taster_twitter_handle 列包含太多的的缺失值
# 
# 所以最终筛选出来的列为: `country, points, province, taster_name, variety`

# In[2]:


# 仅取部分列进行频繁模式挖掘
to_preserve = ['country', 'points', 'province', 'taster_name', 'variety']
data_reduced = data[to_preserve].copy()
data_reduced.info()


# ## 三、对数据集进行处理，以便于进行关联规则挖掘
# 
# 具体处理如下：
# * 将数值属性转化为标称属性
# * 将 DataFrame 格式的数据转化为 List 格式，并删除缺失值
# * 使用预处理工具将数据编码成挖掘工具规定的形式

# In[3]:


# 将数值转化为字符串
data_reduced['points'] = data_reduced['points'].map(str)
data_reduced.info()


# In[4]:


def row_to_list(row):
    return row.dropna().tolist()

data_reduced_list = data_reduced.apply(row_to_list, axis=1).tolist()


# In[5]:


from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
data_encoded = te.fit_transform(data_reduced_list)
df_encoded = pds.DataFrame(data_encoded, columns=te.columns_)
df_encoded.head(3)


# ## 四、使用 mlxtend 工具包找出数据集中的频繁模式

# In[6]:


from mlxtend.frequent_patterns import apriori

freq_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
freq_itemsets.sort_values(by='support', ascending=False, inplace=True)


# In[7]:


print(freq_itemsets[freq_itemsets.itemsets.apply(lambda x: len(x)) >= 1])


# ## 使用 mlxtend 工具包对发现的频繁模式进行关联规则挖掘

# In[8]:


from mlxtend.frequent_patterns import association_rules

asso_rules = association_rules(freq_itemsets, metric='confidence', min_threshold=0.8)
asso_rules.sort_values(by='lift', ascending=False, inplace=True)


# 关联规则挖掘的结果如下，其中第5、6列分别为支持度和置信度，第7、8、9列分别为Lift、Leverage和置信度，都是关联规则的评价指标。

# In[9]:


print(asso_rules)


# 上面展示的是置信度大于等于 0.8 的关联规则，且按照 Lift 值从大到小排序，其中全部都是葡萄酒出产的国家和地区的关联性，这也在意料之内，毕竟地区包含在国家之内，并且一个地区一般只在一个省份内，一个省份也只在一个国家内。
