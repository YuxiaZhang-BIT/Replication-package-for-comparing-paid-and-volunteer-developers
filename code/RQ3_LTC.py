# -*- coding: UTF-8 -*-
from __future__ import division
import os
from datetime import datetime
from statistics import median
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.diagnostic import lilliefors
import pandas as pd

sns.set(style="whitegrid")

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='rust', charset='utf8mb4')
cursor = conn.cursor()

path = os.path.abspath('/Users/Anonymous/Desktop/paid VS volunteer/code/')

data = pd.read_csv(path + '/data/icse24/model_data.csv')
data['gender'] = 'none'
print(data.describe())


id_gender = pd.read_csv(path + '/data/icse24/id_gender_namsor.csv')
print(id_gender.describe())
id_gender = dict(zip(id_gender['Author_ID'], id_gender['gender']))
for index, row in data.iterrows():
    id = row['id']
    if id in id_gender.keys():
        data.at[index, 'gender'] = id_gender[id]


dummy_tasks = pd.get_dummies(data['1st_task'], prefix='task')
print(dummy_tasks)

dummy_paid = pd.get_dummies(data['is_paid'], prefix='paid')
print(dummy_paid)

dummy_gender = pd.get_dummies(data['gender'], prefix='gender')
print(dummy_gender)

# cols_to_keep = ['id', 'status', 'is_paid', 'num_cmt', '1st_loc', '1st_task']
cols_to_keep = ['status', 'num_cmt', '1st_loc']
data = data[cols_to_keep].join(dummy_tasks['task_nonfunctional'])
data = data.join(dummy_tasks['task_corrective'])
data = data.join(dummy_tasks['task_perfective'])
data = data.join(dummy_paid.loc[:, 'paid_1':])
data = data.join(dummy_gender['gender_female'])
data['intercept'] = 1.0
data['num_cmt'] = data['num_cmt'].apply(np.log1p)
data['1st_loc'] = data['1st_loc'].apply(np.log1p)
print(data.head())
print(list(data.columns))

train_cols = data.columns[1:]
logit = sm.Logit(data['status'], data[train_cols])
result = logit.fit()
print(result.summary())
print(result.conf_int())
print(np.exp(result.params))