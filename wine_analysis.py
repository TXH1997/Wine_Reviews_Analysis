#!/usr/bin/env python
# coding: utf-8

# ## 使用pandas读取"Wine Reviews"的数据

# In[1]:


import pandas as pds
data = pds.read_csv('./Wine Reviews/winemag-data-130k-v2.csv')


# ## 标称属性country的聚会频数

# In[2]:


data['country'].value_counts()


# ## 标称属性description的聚会频数

# In[3]:


data['description'].value_counts()


# ## 标称属性desgination的聚会频数

# In[4]:


data['designation'].value_counts()


# ## 标称属性province的聚会频数

# In[5]:


data['province'].value_counts()


# ## 标称属性region_1的聚会频数

# In[6]:


data['region_1'].value_counts()


# ## 标称属性region_2的聚会频数

# In[7]:


data['region_2'].value_counts()


# ## 标称属性taster_name的聚会频数

# In[8]:


data['taster_name'].value_counts()


# ## 标称属性taster_twitter_handle的聚会频数

# In[9]:


data['taster_twitter_handle'].value_counts()


# ## 标称属性title的聚会频数

# In[10]:


data['title'].value_counts()


# ## 标称属性variety的聚会频数

# In[11]:


data['variety'].value_counts()


# ## 标称属性winery的聚会频数

# In[12]:


data['winery'].value_counts()


# ## 数值属性points和price的五数概括

# In[13]:


data.describe()


# ## 数值属性points的缺失值个数

# In[14]:


data['points'].isnull().sum()


# ## 数值属性price的缺失值个数

# In[15]:


data['price'].isnull().sum()


# ## 数值属性points的直方图表示

# In[16]:


data.hist(['points'])


# ## 数值属性points的盒图表示

# In[17]:


data.boxplot(['points'])


# ## 数值属性price的直方图表示

# In[18]:


data.hist(['price'])


# ## 数值属性price的盒图表示

# In[19]:


data.boxplot(['price'])


# ## 剔除缺失值先后price的直方图对比

# In[20]:


dropped_data = data.dropna(subset=['price'])
cmp_dropped_price = pds.DataFrame({'original': data['price'], 'dropped': dropped_data['price']})
cmp_dropped_price.hist()


# ## 剔除缺失值先后price的盒图对比

# In[21]:


cmp_dropped_price.boxplot()


# ## 剔除缺失值先后price的五数概括对比

# In[22]:


cmp_dropped_price.describe()


# ## 以最高频率值填补缺失值，以及填补前后的直方图对比

# In[23]:


most_freq = data['price'].value_counts().index[0]
price_fillna_freq = data['price'].fillna(most_freq)
cmp_fillna_freq_price = pds.DataFrame({'original': data['price'], 'fillna': price_fillna_freq})
cmp_fillna_freq_price.hist()


# ## 盒图对比

# In[24]:


cmp_fillna_freq_price.boxplot()


# ## 五数概括对比

# In[25]:


cmp_fillna_freq_price.describe()


# ## 由于price不符合正态分布，无法使用皮尔逊相关系数通过属性的相关关系来填补缺失值

# In[26]:


import scipy.stats as ss
dropped = data.dropna(subset=['points', 'price'])

ss.normaltest(dropped['price'])


# In[27]:


dropped['price'].corr(dropped['points'], method='pearson')


# ## 计算数据对象相似性的函数，为了减少计算量而只考虑缺失值与附近20个对象的相似性

# In[28]:


def sim(e1, e2):
    score = 0
    score += int(e1['country'] == e2['country'])
    score += int(e1['description'] == e2['description'])
    score += int(e1['designation'] == e2['designation'])
    score += abs(e1['points'] - e2['points'])
    score += int(e1['province'] == e2['province'])
    score += int(e1['region_1'] == e2['region_1'])
    score += int(e1['region_2'] == e2['region_2'])
    score += int(e1['taster_twitter_handle'] == e2['taster_twitter_handle'])
    score += int(e1['taster_name'] == e2['taster_name'])
    score += int(e1['title'] == e2['title'])
    score += int(e1['variety'] == e2['variety'])
    score += int(e1['winery'] == e2['winery'])
    return score
    
def get_fill_value(e0, pos):
    head = pos - 10 if pos - 10 >= 0 else 0
    tail = pos + 10 if pos + 10 <= len(data) else len(data)
    scores = data.loc[range(head, tail)].apply(lambda e: sim(e0, e), axis=1)
    sorted_scores = pds.DataFrame({'score': scores}).sort_values(['score'], ascending=False)
    for i, pos in enumerate(sorted_scores.index.tolist()):
        if i == 0 or data['price'].isnull().values[pos]:
            continue
        else:
            return data['price'].loc[pos]


# ## 使用最相似的数据对象对应的值来填补缺失值

# In[29]:


most_sim = pds.DataFrame({'price': data['price']})
col_num = data['price'][data['price'].isnull().values==True].index
for i in col_num:
    most_sim['price'].loc[i] = get_fill_value(data.loc[i], i)


# ## 填补后的直方图对比

# In[30]:


cmp_fillna_sim_price = pds.DataFrame({'original': data['price'], 'fillna': most_sim['price']})
cmp_fillna_sim_price.hist()


# ## 填补后的盒图对比

# In[31]:


cmp_fillna_sim_price.boxplot()


# ## 填补后的五数概括对比

# In[32]:


cmp_fillna_sim_price.describe()

