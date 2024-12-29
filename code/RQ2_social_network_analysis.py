#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division

import os
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from scipy import stats
from scipy.stats import rankdata

plt.rcParams.update({'font.size': 12})

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='rust', charset='utf8mb4')
cursor = conn.cursor()

path = os.path.abspath('/Users/Yuxia/Desktop/paid VS volunteer/code/')

G = nx.read_gexf(path + '/data/dvpr_file_network.gexf', node_type=str)
print("number of nodes:", G.number_of_nodes())
print("number of edges:", G.number_of_edges())


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


# get collaborations
def get_collaborations(nodelist, label, collaborators):
    for i in nodelist:
        num_paid = 0
        num_vltr = 0
        node = str(i[0]) + ' ' + str(i[1])
        if node in list(G.nodes):
            G.nodes[node]['type'] = label
            neighbers = list(G.neighbors(node))

            for nb in neighbers:
                if nb.split(' ')[1] == 'paid':
                    num_paid += G.get_edge_data(node, nb)['weight']
                if nb.split(' ')[1] == 'vltr':
                    num_vltr += G.get_edge_data(node, nb)['weight']
        collaborators.loc[len(collaborators.index)] = [label, 'Paid developers', num_paid]
        collaborators.loc[len(collaborators.index)] = [label, 'Volunteers', num_vltr]
    return collaborators


# one-time
# get the author_ID of one-time volunteers and paid developers
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, ID_type ' \
          'from icse24 ' \
          'where is_one_time = %s and ID_type = %s'
    cursor.execute(sql, (1, 'paid'))
    one_time_paid = cursor.fetchall()
conn.commit()
# print(one_time_paid)

collaborators = pd.DataFrame(columns=['Group', 'Type', 'Weight'])
collaborators = get_collaborations(one_time_paid, 'One-time', collaborators)


coll_with_paid = collaborators.loc[collaborators['Type'] == 'Paid developers', ['Weight']]
coll_with_vltr = collaborators.loc[collaborators['Type'] == 'Volunteers', ['Weight']]
res = stats.wilcoxon(coll_with_paid, coll_with_vltr, alternative='less')
print('one-time', res)
res = stats.wilcoxon(coll_with_paid, coll_with_vltr)
print('one-time', res)


# peripheral
# get the author_ID of peripheral volunteers and paid developers
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, ID_type ' \
          'from icse24 ' \
          'where is_one_time = %s and ID_type = %s and is_core = %s'
    cursor.execute(sql, (0, 'paid', 0))
    peripheral_paid = cursor.fetchall()
conn.commit()


collaborators = get_collaborations(peripheral_paid, 'Peripheral', collaborators)
res = stats.wilcoxon(coll_with_paid, coll_with_vltr, alternative='less')
print('peripheral', res)
res = stats.wilcoxon(coll_with_paid, coll_with_vltr)
print('peripheral', res)


# core
# get the author_ID of core volunteers and paid developers
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, ID_type ' \
          'from icse24 ' \
          'where ID_type = %s and is_core = %s'
    cursor.execute(sql, ('paid', 1))
    core_paid = cursor.fetchall()
conn.commit()

collaborators = get_collaborations(core_paid, 'Core', collaborators)
res = stats.wilcoxon(coll_with_paid, coll_with_vltr, alternative='less')
print('core', res)
res = stats.wilcoxon(coll_with_paid, coll_with_vltr)
print('core', res)


one_time_paid = collaborators.loc[collaborators['Group'] == 'One-time', ['Type', 'Weight']]
fig, ax = plt.subplots(figsize=(5, 5))
my_pal = {'#D3E5A2', '#ECB5D1'}
sns.boxplot(x="Type", y="Weight", data=one_time_paid, showfliers=False, showmeans=True,
            palette=my_pal, saturation=1)
plt.ylabel("Number of Collaborations", size=12)
plt.xlabel("", size=12)
plt.savefig(path + "/pic/extension/one_time_paid_collaboration.pdf", format='pdf')
plt.show()


peripheral_paid = collaborators.loc[collaborators['Group'] == 'Peripheral', ['Type', 'Weight']]
fig, ax = plt.subplots(figsize=(5, 5))
my_pal = {'#D3E5A2', '#ECB5D1'}
sns.boxplot(x="Type", y="Weight", data=peripheral_paid, showfliers=False, showmeans=True,
            palette=my_pal, saturation=1)
plt.ylabel("Number of Collaborations", size=12)
plt.xlabel("", size=12)
plt.savefig(path + "/pic/extension/peripheral_paid_collaboration.pdf", format='pdf')
plt.show()

core_paid = collaborators.loc[collaborators['Group'] == 'Core', ['Type', 'Weight']]
fig, ax = plt.subplots(figsize=(5, 5))
my_pal = {'#D3E5A2', '#ECB5D1'}
sns.boxplot(x="Type", y="Weight", data=core_paid, showfliers=False, showmeans=True,
            palette=my_pal, saturation=1)
plt.ylabel("Number of Collaborations", size=12)
plt.xlabel("", size=12)
plt.savefig(path + "/pic/extension/core_paid_collaboration.pdf", format='pdf')
plt.show()

print(one_time_paid.loc[one_time_paid['Type'] == 'Paid developers', 'Weight'].median())
print(one_time_paid.loc[one_time_paid['Type'] == 'Volunteers', 'Weight'].median())

print(peripheral_paid.loc[peripheral_paid['Type'] == 'Paid developers', 'Weight'].median())
print(peripheral_paid.loc[peripheral_paid['Type'] == 'Volunteers', 'Weight'].median())

print(core_paid.loc[core_paid['Type'] == 'Paid developers', 'Weight'].median())
print(core_paid.loc[core_paid['Type'] == 'Volunteers', 'Weight'].median())

print(len(one_time_paid.loc[one_time_paid['Type'] == 'Paid developers', 'Weight']))
print(len(one_time_paid.loc[one_time_paid['Type'] == 'Volunteers', 'Weight']))

res = stats.wilcoxon(one_time_paid.loc[one_time_paid['Type'] == 'Paid developers', 'Weight'],
               one_time_paid.loc[one_time_paid['Type'] == 'Volunteers', 'Weight'])
print('one-time', res)

res = stats.wilcoxon(peripheral_paid.loc[peripheral_paid['Type'] == 'Paid developers', 'Weight'],
               peripheral_paid.loc[peripheral_paid['Type'] == 'Volunteers', 'Weight'])
print('peripheral', res)
res = stats.wilcoxon(core_paid.loc[core_paid['Type'] == 'Paid developers', 'Weight'],
               core_paid.loc[core_paid['Type'] == 'Volunteers', 'Weight'])
print('core', res)

collaborators['Weight'] = collaborators['Weight'].astype(float)
collaborators['Weight'] = np.log1p(collaborators['Weight'])
fig, ax = plt.subplots()
my_pal = {'#ECB5D1', '#D3E5A2'}
sns.boxplot(x="Group", y="Weight", hue='Type', data=collaborators, showfliers=False, showmeans=True,
            palette=my_pal, saturation=1)
plt.ylabel("Log-transformed Number of Collaborations", size=12)
plt.xlabel("", size=12)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)
plt.savefig(path + "/pic/extension/collaboration.pdf", format='pdf')
plt.show()
