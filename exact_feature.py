#  -*- encoding: utf-8 -*-
import json
import numpy as np
import tensorflow as tf

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer


from gated_graph_conv import GatedGraphConv
import torch
from torch import Tensor
from torch.nn import Parameter as Param, init
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing

import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from tqdm import tqdm
import jieba
import editdistance
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0"


maxlen = 100  # 160
learning_rate = 5e-5
min_learning_rate = 1e-5

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


'''

'''
def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1 = float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

def convert(sent):
    def double_quote_gen():
        yield u""
        while 1:
            yield u"“"
            yield u"”"

    assert isinstance(sent, unicode)
    seg = sent.split('"')
    if len(seg) % 2 != 1:
        raise RuntimeError('non-balenced quotes!')
    newseg = reduce(tuple.__add__, zip(double_quote_gen(), seg ))
    newsent = reduce(unicode.__add__, newseg)
    return newsent


config_path = '../chinese_wwm_L-12_H-768_A-12/publish/bert_config.json'
checkpoint_path = '../chinese_wwm_L-12_H-768_A-12/publish/bert_model.ckpt'
dict_path = '../chinese_wwm_L-12_H-768_A-12/publish/vocab.txt'


token_dict = {}
with codecs.open(dict_path, 'r', encoding='utf8') as reader:  # dict_path  词典
    for line in reader:
        token = line.strip() #
        token_dict[token] = len(token_dict)

# OurTokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

# one step:read file
def read_data(data_file, table_file):  # data文件，数据表文件
    data, tables = [], {} # 数据，表格
    with open(data_file) as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file) as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']  # column
            d['types'] = l['types']
            d['headers-types'] = list(map(lambda x: x[0]+x[1], zip(d['headers'],d['types'])))
            rows = np.array(l['rows'])
            tables[l['id']] = d
    return data, tables


train_data, train_tables = read_data('../dataset/NL2SQL/train/train.json', '../dataset/NL2SQL/train/train.tables.json')
valid_data, valid_tables = read_data('../dataset/NL2SQL/val/val.json', '../dataset/NL2SQL/val/val.tables.json')
test_data, test_tables = read_data('../dataset/NL2SQL/test/test.json', '../dataset/NL2SQL/test/test.tables.json')
train_data = train_data
valid_data = valid_data
test_data = test_data


bert_model = build_transformer_model(config_path, checkpoint_path)

for l in bert_model.layers:
    l.trainable = True



def achieve_feature1(question, header): # 最新一版本
    global output1
    node = []
    all_node = []
    output2 = []
    x1, x_1 = tokenizer.encode(question)  # x1,x2 代表自然问题编码
    sentence_vec = bert_model.predict([np.array([x1]), np.array([x_1])])[0][0]
    all_node.append(sentence_vec)
    len_header = len(header)
    for j in range(len_header):
        x2, x_2 = tokenizer.encode(header[j])
        column = bert_model.predict([np.array([x2]), np.array([x_2])])[0][0]
        node.append(column)
        all_node.append(column)
    count = 0
    state = []
    for n in range(len(node)):
        if cos_dist(sentence_vec[0], node[n][0]) < 0.8:
            count += 1  # 进行计数
            state.append([0, n])
    # 进行count的补全操作
    if len(state) % 2 == 0:
        state = state
    if len(state) % 2 != 0:
        state.insert(0, [0, 0])
    num_edge_types = len(state) / 2
    if num_edge_types == 1:
        gcn = GatedGraphConv(input_dim=768, num_timesteps=1, num_edge_types=1)
        data = Data(torch.tensor(all_node), edge_index=[
            torch.tensor([state[0], state[1]])
        ])
        output1 = gcn(data.x, data.edge_index)
    if num_edge_types == 2:
        gcn = GatedGraphConv(input_dim=768, num_timesteps=2, num_edge_types=2)
        data = Data(torch.tensor(all_node), edge_index=[
            torch.tensor([state[0], state[1]]),
            torch.tensor([state[2], state[3]])
        ])
        output1 = gcn(data.x, data.edge_index)


    # output2 = output2.append(output1.cpu().detach().numpy())

    return output1


# 进行循环训练集 　获取对应的的张量 data 对应train_data, tables 对应train_tables
def achieve_feature(Nl2SQL_data, Nl2SQL_tables):
    output2 = list()
    # output1 = list()
    global output1
    node = []
    all_node = []
    for i in range(len(Nl2SQL_data)): # 要不要循环date
        # global output1
        # global output2
        # print(train_data[i])
        # print(data[i])

        x1, x_1 = tokenizer.encode(str(Nl2SQL_data[i]['question']))  # x1,x2 代表自然问题编码
        sentence_vec = bert_model.predict([np.array([x1]), np.array([x_1])])[0][0]
        all_node.append(sentence_vec)
        len_header = len(Nl2SQL_tables[Nl2SQL_data[i]['table_id']]['headers-types'])
        for j in range(len_header):
            x2, x_2 = tokenizer.encode(Nl2SQL_tables[Nl2SQL_data[i]['table_id']]['headers-types'][j])
            column = bert_model.predict([np.array([x2]), np.array([x_2])])[0][0]
            node.append(column)
            all_node.append(column)
        count = 0
        state = []
        for n in range(len(node)):
            if cos_dist(sentence_vec[0], node[n][0]) < 0.8:
                count += 1  # 进行计数
                state.append([0, n])
        # 进行count的补全操作
        if len(state) % 2 == 0:
            state = state
        if len(state) % 2 != 0:
            state.insert(0, [0, 0])
        num_edge_types = len(state)/2
        if num_edge_types == 1:
            gcn = GatedGraphConv(input_dim=768, num_timesteps=1, num_edge_types=1)
            data = Data(torch.tensor(all_node), edge_index=[
                torch.tensor([state[0],state[1]])
            ])
            output1 = gcn(data.x, data.edge_index)
        if num_edge_types == 2:
            gcn = GatedGraphConv(input_dim=768, num_timesteps=2, num_edge_types=2)
            data = Data(torch.tensor(all_node), edge_index=[
                torch.tensor([state[0], state[1]]),
                torch.tensor([state[2], state[3]])
            ])
            output1 = gcn(data.x, data.edge_index)
        # output1 = output1.cpu().detach().numpy()
        output2.append(output1)

    return output2

'''
x1, x_1 = tokenizer.encode("哪个科目的教材是人民教育出版社出版的") # x1,x2 代表自然问题编码
x2,x_2 = tokenizer.encode("年级")
x3,x_3 = tokenizer.encode("学科")
x4,x_4 = tokenizer.encode("教材出版社")
x5,x_5 = tokenizer.encode("教辅名称")
x6,x_6 = tokenizer.encode("教辅出版社")
x7,x_7 = tokenizer.encode("定价")


sentence_vec = bert_model.predict([np.array([x1]), np.array([x_1])])[0][0]
column_1 = bert_model.predict([np.array([x2]), np.array([x_2])])[0][0]
column_2 = bert_model.predict([np.array([x3]), np.array([x_3])])[0][0]
column_3 = bert_model.predict([np.array([x4]), np.array([x_4])])[0][0]
column_4 = bert_model.predict([np.array([x5]), np.array([x_5])])[0][0]
column_5 = bert_model.predict([np.array([x6]), np.array([x_6])])[0][0]
column_6 = bert_model.predict([np.array([x7]), np.array([x_7])])[0][0]
node.append(column_1)
node.append(column_2)
node.append(column_3)
node.append(column_4)
node.append(column_5)
node.append(column_6)

all_node.append(sentence_vec)
all_node.append(column_1)
all_node.append(column_2)
all_node.append(column_3)
all_node.append(column_4)
all_node.append(column_5)
all_node.append(column_6)

count = 0
state = []
for i in range(len(node)):
    if cos_dist(sentence_vec[0], node[i][0]) < 0.8:
        count += 1  # 进行计数
        state.append([0, i])
print(count)
print(state)
# 进行count的补全操作
if len(state) % 2 == 0:
    state = state
if len(state) % 2 != 0:
    state.insert(0, [0,0])


# print(torch.zeros((5,10)))
# print(torch.tensor(all_node))

gcn = GatedGraphConv(input_dim=768, num_timesteps=1, num_edge_types=1)
data = Data(torch.tensor(all_node), edge_index=[
        torch.tensor([[0,1],[2,3]])
    ])
output1 = gcn(data.x, data.edge_index)

print(output1)

'''










