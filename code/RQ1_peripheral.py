# -*- coding: UTF-8 -*-
from __future__ import division
import pymysql
import os
from datetime import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors
from scipy import stats
import seaborn as sns
from scipy.stats import tiecorrect, rankdata, norm
from statistics import median
from statistics import mean
import pandas as pd
import math
plt.rcParams.update({'font.size': 12})
# sns.set(style="whitegrid")

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='rust', charset='utf8mb4')
cursor = conn.cursor()

path = os.path.abspath('/Users/Yuxia/Desktop/paid VS volunteer/code/')

# get affiliations of developers
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, company ' \
          'from icse24 ' \
          'where company != %s and company != %s ' \
          'and is_one_time = %s and is_core = %s'
    cursor.execute(sql, ('unknown', 'bot', 0, 0))
    res = cursor.fetchall()
    print(res)
conn.commit()

dict_dvpr_affi = {}
for i in res:
    if i[0] in dict_dvpr_affi.keys():
        dict_dvpr_affi[i[0]].append(i[1])
    else:
        dict_dvpr_affi[i[0]] = [i[1]]


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


# get the median value
def get_median(data):
    locs = []
    dvpr_loc = []
    index = 0
    # id, author_ID, sum_lines
    for i in data:
        a_id = i[1]
        loc = i[2]
        # print(i)
        if a_id != index and index != 0:
            # print('author', index)
            # print('dvpr_loc', dvpr_loc)
            locs.append(median(dvpr_loc))
            dvpr_loc = []
        dvpr_loc.append(loc)
        index = a_id
    locs.append(median(dvpr_loc))
    return locs


print('***************** LOC *********************')
# get loc in the commits contributed by volunteers
with conn.cursor() as cursor:
    sql = 'select id, author_ID, sum_lines ' \
          'from icse24 ' \
          'where company = %s ' \
          'and is_one_time = %s and is_core = %s ' \
          'order by author_ID'
    cursor.execute(sql, ('volunteer', 0, 0))
    res = cursor.fetchall()
    # print(res)
conn.commit()
loc_vltr = get_median(res)

# get loc in the commits contributed by paid developers
with conn.cursor() as cursor:
    sql = 'select id, author_ID, sum_lines ' \
          'from icse24 ' \
          'where company != %s and company != %s and company != %s ' \
          'and is_one_time = %s and is_core = %s ' \
          'order by author_ID'
    cursor.execute(sql, ('volunteer', 'unknown', 'bot', 0, 0))
    res = cursor.fetchall()
conn.commit()
loc_coms = get_median(res)


# test the difference of LOCs between volunteers and paid-developers
fig, ax = plt.subplots()
ax.boxplot([loc_vltr, loc_coms], showfliers=False, showmeans=True)
ax.set_xticklabels(["Volunteers", "Paid Developers"])
ax.set_ylabel('Line of Code')
#plt.savefig(path + "/pic/icse24/loc.pdf", format='pdf')
plt.show()

print(lilliefors(loc_vltr))
print(lilliefors(loc_coms))
# both p-value < 0.05, indicating the two distributions are not norm. Thus we choose Mann-Whitney U
uStat, pVal = stats.mannwhitneyu(loc_vltr, loc_coms)
if pVal < 0.05:
    print("两组数据有统计学差异", pVal)
else:
    print("两组数据没有统计学差异", pVal)
print('effect size', get_effectsize(loc_vltr, loc_coms))
print('number', len(loc_vltr), len(loc_coms))

print('loc')
print('median_vltr', median(loc_vltr))
print('median_com', median(loc_coms))
print('mean_vltr', mean(loc_vltr))
print('mean_com', mean(loc_coms))

print(len(loc_coms))

role_type_loc = pd.read_csv(path + "/data/icse24/role_type_loc.csv")
for i in loc_vltr:
    role = 'Peripheral'
    type = 'volunteer'
    loc = i
    role_type_loc.loc[len(role_type_loc.index)] = [role, type, loc]

for i in loc_coms:
    role = 'Peripheral'
    type = 'paid'
    loc = i
    role_type_loc.loc[len(role_type_loc.index)] = [role, type, loc]
role_type_loc.to_csv(path + "/data/icse24/role_type_loc.csv", index=False)


# contribution frequency
print('***************** contribution frequency *********************')
# get start date, end date, and sum of contributed commits of developers
with conn.cursor() as cursor:
    sql = 'select author_ID, company, min(author_date), max(author_date), count(*) ' \
          'from icse24 ' \
          'where company != %s and company != %s ' \
          'and is_one_time = %s and is_core = %s ' \
          'group by author_ID, company ' \
          'order by author_ID'
    cursor.execute(sql, ('unknown', 'bot', 0, 0))
    res = cursor.fetchall()
conn.commit()
print(len(res))


# calculate contribution frequency
def get_frequency(data):
    vltr_freq = []
    paid_freq = []
    # author_ID, company, min(author_date), max(author_date), count(*)
    for i in data:
        st = i[2]
        ed = i[3]
        num_months = (ed - st).days / 30
        if num_months == 0:
            num_months = 1
        # print(i, num_months)
        sum_cmt = i[4]
        freq = sum_cmt / num_months

        if i[1] == 'volunteer':
            vltr_freq.append(freq)
        else:
            paid_freq.append(freq)

    return vltr_freq, paid_freq


vltr_freq, com_freq = get_frequency(res)
print(lilliefors(vltr_freq))
print(lilliefors(com_freq))
print('median_vltr', median(vltr_freq))
print('median_com', median(com_freq))
# both p-value < 0.05, indicating the two distributions are not norm. Thus we choose Mann-Whitney U
uStat, pVal = stats.mannwhitneyu(vltr_freq, com_freq)
if pVal < 0.05:
    print("两组数据有统计学差异", pVal)
else:
    print("两组数据没有统计学差异", pVal)
print('effect size', get_effectsize(vltr_freq, com_freq))
print('number', len(vltr_freq), len(com_freq))
# np.savetxt(path + "/data/icse24/peri_frequency.csv", id_interval, fmt="%s", delimiter=",")

df_freq = pd.DataFrame(columns=['Type', 'Frequency'], dtype=float)
for i in vltr_freq:
    df_freq.loc[len(df_freq.index)] = ['Volunteer', i]
for i in com_freq:
    df_freq.loc[len(df_freq.index)] = ['Paid', i]

plt.figure(figsize=(6, 4))
sns.boxplot(x=df_freq['Type'], y=df_freq['Frequency'], palette="Set2", showfliers=False)
plt.xlabel('')
plt.ylabel('Frequency')
# plt.savefig(path + "/pic/icse24/loc.pdf", format='pdf')
plt.show()


role_type_freq = pd.DataFrame(columns=['Role', 'Type', 'Contribution Frequency'])
for index, row in df_freq.iterrows():
    role = 'Peripheral'
    type = row['Type']
    freq = row['Frequency']
    role_type_freq.loc[len(role_type_freq.index)] = [role, type, freq]
role_type_freq.to_csv(path + "/data/icse24/role_type_freq.csv", index=False)


# **** task preference ****
print('***************** task preference *********************')
# get affiliations, task, and task number of developers
with conn.cursor() as cursor:
    sql = 'select author_ID, company, task_type, count(task_type) ' \
          'from icse24 ' \
          'where company != %s and company != %s ' \
          'and is_one_time = %s and is_core = %s ' \
          'group by author_ID, company, task_type ' \
          'order by author_ID, company, task_type'
    cursor.execute(sql, ('unknown', 'bot', 0, 0))
    res = cursor.fetchall()
    # print(res)
conn.commit()
# print(len(res)) 5631


# task perfective, features, corrective, nonfunctional]
df_dvpr_task = pd.DataFrame(columns=['author_ID', 'affiliation_type', 'affiliation', 'perfective', 'features',
                                     'corrective', 'nonfunctional'])
for i in res:
    a_id = i[0]
    affi = i[1]
    if affi == 'volunteer':
        affi_type = 'Volunteer'
    else:
        affi_type = 'Paid developer'
    if a_id not in df_dvpr_task['author_ID'].values.tolist() or affi not in \
            df_dvpr_task.loc[(df_dvpr_task.author_ID == a_id), 'affiliation'].values.tolist():
        df_dvpr_task.loc[len(df_dvpr_task.index)] = [a_id, affi_type, affi, 0, 0, 0, 0]

    task = i[2]
    num = i[3]
    if task != 'unknown':
        df_dvpr_task.loc[(df_dvpr_task.affiliation == affi) & (df_dvpr_task.author_ID == a_id), task] = num

print(df_dvpr_task)
df_dvpr_task.to_csv(path + '/data/icse24/dvpr_task.csv')

task_ratio = pd.DataFrame(columns=['Role', 'task', 'ratio'])
paid_task = pd.DataFrame(columns=['perfective', 'features', 'corrective', 'nonfunctional'])
vltr_task = pd.DataFrame(columns=['perfective', 'features', 'corrective', 'nonfunctional'])
for index, row in df_dvpr_task.iterrows():
    sum_cmt = row['perfective'] + row['features'] + row['corrective'] + row['nonfunctional']
    if sum_cmt > 5:
        r_perfective = row['perfective'] / sum_cmt
        r_feature = row['features'] / sum_cmt
        r_corrective = row['corrective'] / sum_cmt
        r_nonfunctional = row['nonfunctional'] / sum_cmt
        role = row['affiliation_type']

        task_ratio.loc[len(task_ratio.index)] = [role, 'Perfective', r_perfective]
        task_ratio.loc[len(task_ratio.index)] = [role, 'Feature', r_feature]
        task_ratio.loc[len(task_ratio.index)] = [role, 'Corrective', r_corrective]
        task_ratio.loc[len(task_ratio.index)] = [role, 'Nonfunctional', r_nonfunctional]

        if row['affiliation_type'] == 'Volunteer':
            vltr_task.loc[len(vltr_task.index)] = [r_perfective, r_feature, r_corrective, r_nonfunctional]
        else:
            paid_task.loc[len(paid_task.index)] = [r_perfective, r_feature, r_corrective, r_nonfunctional]

task_ratio.to_csv(path + '/data/icse24/peripheral_task_ratio.csv')

# visualization
fig, ax = plt.subplots()
sns.boxplot(x="task", y="ratio", hue='Role', data=task_ratio, showfliers=False, showmeans=True)
plt.ylabel("Percentage", size=12)
plt.xlabel("Commit Categories", size=12)
plt.savefig(path + "/pic/icse24/tasks.pdf", format='pdf')
plt.show()


fig, ax = plt.subplots(figsize=(7, 4))
my_pal = {'#ECB5D1', '#D3E5A2'}
sns.boxplot(x="task", y="ratio", hue='Role', data=task_ratio, showfliers=False, showmeans=True,
            palette=my_pal, saturation=1)
plt.ylabel("Task Ratio")
plt.xlabel("", size=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=('Volunteer', 'Paid'))
plt.savefig(path + "/pic/icse24/peripheral_task.pdf", format='pdf')
plt.show()

print(task_ratio)
data = pd.DataFrame(columns=['Type', 'Ratio'])
for index, row in task_ratio.iterrows():
    if row['Role'] == 'Volunteer' and row['task'] == 'Feature':
        data.loc[len(data.index)] = ['Volunteer', row['ratio']]
    if row['Role'] == 'Paid developer' and row['task'] == 'Feature':
        data.loc[len(data.index)] = ['Paid developer', row['ratio']]

print('88888', len(data))
my_pal = {"volunteer": "b", "paid developer": "m"}
plt.figure(figsize=(6, 4))
sns.violinplot(x=data['Type'], y=data['Ratio'], palette="Set2", split=True, scale="width")
plt.xlabel('')
plt.savefig(path + "/pic/test_feature.pdf", format='pdf')
plt.show()



# compare differences
def compare_difference(data1, data2, feature):
    print('**************', feature)
    p_value1 = lilliefors(data1)[1]
    p_value2 = lilliefors(data2)[1]
    print(p_value1, p_value2)
    if p_value1 < 0.05 or p_value2 < 0.05:
        uStat, pVal = stats.mannwhitneyu(data1, data2)
        if pVal < 0.05:
            print(feature, ": paid developers与volunteer有统计学差异", pVal)
        else:
            print(feature, ": paid developers与volunteer没有统计学差异", pVal)
        print('effect size', get_effectsize(data1, data2))


# compare difference
print(vltr_task['perfective'], paid_task['perfective'])
compare_difference(vltr_task['perfective'], paid_task['perfective'], 'perfective')
compare_difference(vltr_task['features'], paid_task['features'], 'features')
compare_difference(vltr_task['corrective'], paid_task['corrective'], 'corrective')
compare_difference(vltr_task['nonfunctional'], paid_task['nonfunctional'], 'nonfunctional')
compare_difference(vltr_task['nonfunctional'] + vltr_task['perfective'],
                   paid_task['nonfunctional'] + paid_task['perfective'], 'merge')

print(median(vltr_task['perfective']), median(paid_task['perfective']))
print(median(vltr_task['features']), median(paid_task['features']))
print(median(vltr_task['corrective']), median(paid_task['corrective']))
print(median(vltr_task['nonfunctional']), median(paid_task['nonfunctional']))
