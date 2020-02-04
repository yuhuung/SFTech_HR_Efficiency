#!/usr/bin/env python
# coding: utf-8

# In[5]:


import random
import numpy as np
import scipy.special #for the signmoid function expit()
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# In[6]:


rawdata = pd.read_excel('X1_ver4_cleaned.xlsx')
# 分类变量转化为值变量
data1 = rawdata[['t1.tickets', 't1.avg_pce_cnt', 't1.pce_tickets', 't1.days',
       't1.accumulate_tickets', 't1.accumulate_avg_pce_cnt',
       't1.accumulate_pce_tickets', 't1.last_mon_tickets',
       't1.last_mon_avg_pce_cnt', 't1.last_mon_pce_tickets', 't1.y_days',
       't2.dept_type_name', 't2.dept_type_level',
       't2.dept_manage_flag', 't2.longitude', 't2.latitude',
       't2.belong_county','live_tm']]
print(rawdata['t2.dept_type_name'].unique())
print(rawdata['t2.belong_county'].unique())
pf1 = pd.get_dummies(data1[['t2.dept_type_name']])
df1 = pd.concat([data1, pf1], axis=1)
df1.drop(['t2.dept_type_name'], axis=1, inplace=True)
pf2 = pd.get_dummies(df1[['t2.belong_county']])
df2 = pd.concat([df1, pf2], axis=1)
df2.drop(['t2.belong_county'], axis=1, inplace=True)
# y放第一位，累积量减去本月变为历史累积量
df3 = df2[['t1.tickets', 't1.avg_pce_cnt', 't1.pce_tickets', 
       't1.accumulate_tickets', 't1.last_mon_tickets',
       't1.last_mon_avg_pce_cnt', 't1.last_mon_pce_tickets', 
       't2.dept_type_level', 't2.dept_manage_flag', 't2.longitude',
       't2.latitude', 'live_tm', 't2.dept_type_name_营业点',
       't2.dept_type_name_营业站', 't2.dept_type_name_营业部',
       't2.dept_type_name_集收客户营业部', 
       't2.belong_county_光明区', 't2.belong_county_光明新区', 't2.belong_county_南山区',
       't2.belong_county_坪山区', 't2.belong_county_坪山新区',
       't2.belong_county_大鹏新区', 't2.belong_county_宝安区', 't2.belong_county_盐田区',
       't2.belong_county_福田区', 't2.belong_county_罗湖区', 't2.belong_county_龙华区',
       't2.belong_county_龙华新区', 't2.belong_county_龙岗区']]
df3['t1.accumulate_tickets'] = df3['t1.accumulate_tickets'] - df3['t1.tickets']


# In[31]:


#分为训练集（60%）和测试集(40%)
input_data = df3.iloc[:,1:]
output_data= df3.iloc[:,0]

def divided(xdata,ydata,percent=0.4):
    x_train, x_test, y_train, y_test = train_test_split(input_data,output_data,test_size=percent, random_state=0)#导入数据

    #scale
    x_train=x_train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) *0.99 + 0.01)
    x_test =x_test. apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) *0.99 + 0.01)
    y_train=pd.DataFrame(y_train).apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) *0.99)
    y_test =pd.DataFrame(y_test). apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) *0.99)
    
    return np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)


# In[32]:


ymax = np.max(output_data.values, keepdims=True)
ymin = np.min(output_data.values, keepdims=True)
model_data = list(divided(input_data,output_data))
model_data.append([ymax,ymin])

