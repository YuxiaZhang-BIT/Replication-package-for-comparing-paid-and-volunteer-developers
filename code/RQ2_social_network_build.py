#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division

import os

import networkx as nx
import pymysql
import seaborn as sns

sns.set(style="whitegrid")

conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='123456', db='rust', charset='utf8mb4')
cursor = conn.cursor()

path = os.path.abspath('/Users/Yuxia/Desktop/paid VS volunteer/code/')


# get the list of developers' touched files in different groups
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, ID_type, date_format(author_date, %s), file ' \
          'from icse24, file_changed ' \
          'where ID_type != %s and icse24.rev = file_changed.rev'
    cursor.execute(sql, ('%Y-%m', ''))
    dvprs_files = cursor.fetchall()
conn.commit()

dict_dvpr_files = {}
for i in dvprs_files:
    print(i)
    key = str(i[0]) + ' ' + str(i[1])
    mon = i[2]
    if key in dict_dvpr_files.keys():
        if mon in dict_dvpr_files[key].keys():
            dict_dvpr_files[key][mon].append(i[3])
        else:
            dict_dvpr_files[key][mon] = [i[3]]
    else:
        dict_dvpr_files[key] = {mon: [i[3]]}

print(dict_dvpr_files)


def get_id_type(node):
    id = int(node.split(' ')[0])
    id_type = node.split(' ')[1]
    return id, id_type


# get the number of files two developers both contributed within one month
def get_weight(node_a, node_b):
    a_month_files = dict_dvpr_files[node_a]
    b_month_files = dict_dvpr_files[node_b]

    weight = 0
    for ka in a_month_files.keys():
        if ka in b_month_files.keys():
            common = list(set(a_month_files[ka]).intersection(set(b_month_files[ka])))
            weight += len(common)

    return weight


# get developers (treat transfer dvprs as two identities)
with conn.cursor() as cursor:
    sql = 'select distinct author_ID, ID_type ' \
          'from icse24 ' \
          'where company != %s and company != %s ' \
          'order by author_ID'
    cursor.execute(sql, ('unknown', 'bot'))
    dvprs = cursor.fetchall()
conn.commit()

# add nodes
nodes = []
for i in dvprs:
    nodes.append(str(i[0]) + ' ' + str(i[1]))
print(len(nodes))

# calculate edges and weights
edges = []  # node A, weight, node B
for i in range(0, len(nodes) - 1):
    node_a, a_type = get_id_type(nodes[i])
    for j in range(1, len(nodes)):
        node_b, b_type = get_id_type(nodes[j])
        if node_a == node_b:
            continue
        weight_ab = get_weight(nodes[i], nodes[j])
        if weight_ab != 0:
            edges.append([nodes[i], weight_ab, nodes[j]])
        print(nodes[i], weight_ab, nodes[j])
    print('********** ', i, ' *************')
    break

G = nx.Graph()
for e in edges:
    weight = e[1]
    if weight > 0:
        G.add_edge(e[0], e[2], weight=weight)

print(G.number_of_nodes())
print(G.number_of_edges())
nx.write_gexf(G, path + '/data/dvpr_file_network.gexf')

