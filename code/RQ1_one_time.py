# -*- coding: UTF-8 -*-
from __future__ import division
import os
from datetime import datetime
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
import seaborn as sns
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.diagnostic import lilliefors

# sns.set(style="whitegrid")

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='rust', charset='utf8mb4')
cursor = conn.cursor()

path = os.path.abspath('/Users/Yuxia/Desktop/paid VS volunteer/code/')


def get_effectsize(x, y):
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y

    # use the min(u1, u2) as u-stat
    if u1 <= u2:
        stat_a, larger = u1, 1
    else:
        stat_a, larger = u2, 2

    # compute the effect size
    effect = 1 - (2 * stat_a) / (n1 * n2)
    return effect


# whether the developer is a newcomer
# get the sum of commits contributed by each developers
with conn.cursor() as cursor:
    sql = 'select author_ID, count(*) ' \
          'from icse24 ' \
          'where company != %s and company != %s '\
          'group by author_ID ' \
          'order by count(*)'
    cursor.execute(sql, ('unknown', 'bot'))
    res = cursor.fetchall()
    print(res)
conn.commit()

newcomer_ids = []
for i in res:
    if i[1] < 2:
        newcomer_ids.append(i[0])
    else:
        break

# get affiliations of developers
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, company ' \
          'from icse24 ' \
          'where company != %s and company != %s'
    cursor.execute(sql, ('unknown', 'bot'))
    res = cursor.fetchall()
conn.commit()
dict_dvpr_affi = {}
for i in res:
    if i[0] in dict_dvpr_affi.keys():
        dict_dvpr_affi[i[0]].append(i[1])
    else:
        dict_dvpr_affi[i[0]] = [i[1]]

# the distribution of paid and voluntary newcomers
paid_nc = []
vltr_nc = []
# questionable_id = []
for i in newcomer_ids:
    affi = dict_dvpr_affi[i]
    if len(affi) == 1:
        if affi[0] == 'volunteer':
            vltr_nc.append(i)
        else:
            paid_nc.append(i)

print(len(newcomer_ids))  # 1897
print(len(paid_nc))  # 91
print(len(vltr_nc))  # 1787

print(paid_nc)
print(vltr_nc)


# get loc and number of changed files in the commits contributed by developers
with conn.cursor() as cursor:
    sql = 'select id, author_ID, company, sum_lines, num_files, task_type ' \
          'from icse24 ' \
          'where company != %s and company != %s ' \
          'order by author_ID'
    cursor.execute(sql, ('unknown', 'bot'))
    res = cursor.fetchall()
conn.commit()

data_newcomer = pd.DataFrame(columns=['author_ID', 'Type', 'Company', 'Line of Code', 'n_files', 'task_type'], dtype=float)
for i in res:
    a_id = i[1]
    if a_id in paid_nc:
        affi = 'paid'
        data_newcomer.loc[len(data_newcomer.index)] = [a_id, affi, i[2], i[3], i[4], i[5]]
    elif a_id in vltr_nc:
        affi = 'volunteer'
        data_newcomer.loc[len(data_newcomer.index)] = [a_id, affi, i[2], i[3], i[4], i[5]]
    else:
        continue

print(data_newcomer)
#data_newcomer.to_csv(path + "/data/icse24/data_newcomer.csv")


my_pal = {"volunteers": "b", "paid developers": "m"}
plt.figure(figsize=(6, 4))
sns.boxplot(x=data_newcomer['Type'], y=data_newcomer['Line of Code'], palette="Set2", showfliers=False)
plt.xlabel('')
plt.ylabel('Line of Code')
#plt.savefig(path + "/pic/icse24/loc.pdf", format='pdf')
plt.show()


paid_nc_loc = data_newcomer.loc[data_newcomer['Type'] == 'paid', ['Line of Code']]
vltr_nc_loc = data_newcomer.loc[data_newcomer['Type'] == 'volunteer', ['Line of Code']]
print(lilliefors(paid_nc_loc['Line of Code']))
print(lilliefors(vltr_nc_loc['Line of Code']))
# both p-value < 0.05, indicating the two distributions are not norm. Thus we choose Mann-Whitney U
uStat, pVal = stats.mannwhitneyu(paid_nc_loc['Line of Code'], vltr_nc_loc['Line of Code'])
if pVal < 0.05:
    print("两组数据有统计学差异", pVal)
else:
    print("两组数据没有统计学差异", pVal)
print('effect size', get_effectsize(paid_nc_loc['Line of Code'], vltr_nc_loc['Line of Code']))
print('median paid loc', median(paid_nc_loc['Line of Code']))
print('median vltr loc', median(vltr_nc_loc['Line of Code']))


paid_tasks = data_newcomer.loc[data_newcomer['Type'] == 'paid', ['task_type']]
vltr_tasks = data_newcomer.loc[data_newcomer['Type'] == 'volunteer', ['task_type']]
print('The number of different tasks conducted by paid newcomers:')
print(paid_tasks.value_counts())
print('The number of different tasks conducted by voluntary newcomers:')
print(vltr_tasks.value_counts())

role_type_loc = pd.DataFrame(columns=['Role', 'Type', 'Line of Code'])
for index, row in data_newcomer.iterrows():
    role = 'One-time'
    type = row['Type']
    loc = row['Line of Code']
    role_type_loc.loc[len(role_type_loc.index)] = [role, type, loc]
role_type_loc.to_csv(path + "/data/icse24/role_type_loc.csv", index=False)

