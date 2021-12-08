#! -*- coding: utf-8 -*-

import json
import numpy as np
import numpy
import tensorflow as tf

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

import codecs
from keras.layers import *
from keras.layers import Concatenate
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from tqdm import tqdm
import matplotlib.pyplot as plt
import jieba
import editdistance
import re
import os
os.environ['CUDA_VISIBLE_DEVICES'] ="0"
from question_perpor import trans_question_acc


from exact_feature import achieve_feature, achieve_feature1


maxlen = 160
learning_rate = 5e-5
min_learning_rate = 1e-5


num_agg = 7 # agg_sql_dict = {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
num_op = 5 # {0:">", 1:"<", 2:"==", 3:"!=", 4:"不被select"}
num_cond_conn_op = 3 # conn_sql_dict = {0:"", 1:"and", 2:"or"}



config_path = '../chinese_wwm_L-12_H-768_A-12/publish/bert_config.json'
checkpoint_path = '../chinese_wwm_L-12_H-768_A-12/publish/bert_model.ckpt'
dict_path = '../chinese_wwm_L-12_H-768_A-12/publish/vocab.txt'

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
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['types'] = l['types']
            d['headers-types'] = list(map(lambda x: x[0] + x[1], zip(d['headers'], d['types'])))
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables


train_data, train_tables = read_data('../dataset/NL2SQL/train/train.json', '../dataset/NL2SQL/train/train.tables.json')
valid_data, valid_tables = read_data('../dataset/NL2SQL/val/val.json', '../dataset/NL2SQL/val/val.tables.json')
test_data, test_tables = read_data('../dataset/NL2SQL/test/test.json', '../dataset/NL2SQL/test/test.tables.json')
train_data = train_data[:100]
valid_data = valid_data[:100]
test_data = test_data[:100]



token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader: # dict_path  词典
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


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])

# 从词表中找最相近的词（当无法全匹配的时候）
def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]

# 从句子s中找与w最相近的片段，借助分词工具和ngram的方式尽量精确地确定边界。
def most_similar_2(w, s):  #（，句子）
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)


# data generate
class data_generator:
    def __init__(self, data, tables, achieve_feature, batch_size=8): # 初始化
        self.data = data
        self.tables = tables
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        self.achieve_feature = achieve_feature  # 加不加參數
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self): # 每次迭代
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            nighbour_node = self.achieve_feature
            X1, X2, X3, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                d = self.data[i] # d
                t = self.tables[d['table_id']]['headers']  # 数据表header
                d['question'] = trans_question_acc(d['question'])
                x1, x2 = tokenizer.encode(d['question']) # x1,x2 代表自然问题编码

                # x3 = np.zeros((len(t)+1, 768), dtype='float')
                # x3 = np.array(achieve_feature(self.data, self.tables)[0])  # (len(t)+1, 768)
                x3 = np.array(nighbour_node[i][0].detach().numpy())  # (1, 768)
                # print("x3.shape is")
                # print(x3.shape)

                xm = [0] + [1] * len(d['question']) + [0]  # x 的mask 表示
                h = [] # 列名
                for j in t: # j 来自table
                    _x1, _x2 = tokenizer.encode(j) # 关于数据表头表示编码
                    h.append(len(x1))
                    x1.extend(_x1)  # 添加数据表头编码 indexs
                    x2.extend(_x2) # 添加数据表头编码segemnt
                hm = [1] * len(h)  # 列名mask
                sel = [] # 被select的列 {count, min, max}
                for j in range(len(h)):  # 列名的长度 for循环根据h的长度
                    if j in d['sql']['sel']: # "sel": [7]
                        j = d['sql']['sel'].index(j)
                        sel.append(d['sql']['agg'][j]) # 添加编号{count,min,max} "agg": [0]
                    else:
                        sel.append(num_agg - 1) # 不被select则被标记为num_agg-1
                conn = [d['sql']['cond_conn_op']]  #  连接类型 # "cond_conn_op": 0
                csel = np.zeros(len(d['question']) + 2, dtype='int32') #  被选择的列，这里的0既表示padding，又表示第一列，padding部分训练时会被mask
                cop = np.zeros(len(d['question']) + 2, dtype='int32') + num_op - 1 #  条件中的运算符（同时也是值的标记） 不被select则被标记为num_op-1
                for j in d['sql']['conds']: # "conds": [[1, 2, "捷成股份"]]  j  来自d['sql']['conds']
                    if j[2] not in d['question']:
                        j[2] = most_similar_2(j[2], d['question'])  # (，问题编码)
                    if j[2] not in d['question']:
                        continue
                    k = d['question'].index(j[2]) # k 值代表初始位置
                    csel[k + 1: k + 1 + len(j[2])] = j[0] # 问题中列名[1,1,0,0,0,0]
                    cop[k + 1: k + 1 + len(j[2])] = j[1]  # 运算符号[0,0,0,0,1,1]
                if len(x1) > maxlen:
                    continue
                X1.append(x1)  # bert的输入 其中包含列名
                X2.append(x2)  # bert的输入　其中包含列名
                # print('X1 len is', len(X1))
                X3.append(x3)
                # print('X3 len is',len(X3))
                XM.append(xm) # 输入序列的mask
                H.append(h) # 列名所在位置
                HM.append(hm) # 列名mask
                SEL.append(sel) # 被select的列
                CONN.append(conn) # 连接类型
                CSEL.append(csel) # 条件中的列
                COP.append(cop) # 条件中的运算符（同时也是值的标记）
                if len(X1) == self.batch_size:
                    X1 = seq_padding(X1)
                    # print(X1)
                    X2 = seq_padding(X2)
                    # print(X2)
                    X3 = seq_padding(X3)
                    # print(X3.shape)
                    XM = seq_padding(XM, maxlen=X1.shape[1])
                    # print(XM)
                    H = seq_padding(H)
                    # print(H)
                    HM = seq_padding(HM)
                    # print(HM)
                    SEL = seq_padding(SEL)
                    # print(SEL)
                    CONN = seq_padding(CONN)
                    # print(CONN)
                    CSEL = seq_padding(CSEL, maxlen=X1.shape[1])
                    # print(CSEL)
                    COP = seq_padding(COP, maxlen=X1.shape[1])
                    # print(COP)
                    yield [X1, X2, X3, XM, H, HM, SEL, CONN, CSEL, COP], None
                    X1, X2, X3, XM, H, HM, SEL, CONN, CSEL, COP = [], [], [], [], [], [], [], [], [], []


def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, n]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, n, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    return K.tf.batch_gather(seq, idxs)



# 上述是模块的介绍，接下来是利用模块
# 加载模块
bert_model = build_transformer_model(config_path, checkpoint_path)

for l in bert_model.layers:
    l.trainable = True


x1_in = Input(shape=(None,))  # bert的问题文本输入 shape=(?,?)
x2_in = Input(shape=(None,))  # bert的问题文本输入 shape=(?,?)

x3_in = Input(shape=(768,))  #[?,768]
# print(x3_in)

xm_in = Input(shape=(None,)) # 输入序列的mask  shape=(?,?)
h_in = Input(shape=(None,), dtype='int32') # 列名所在位置  shape=(?,?)
hm_in = Input(shape=(None,))  # 列名mask  shape=(?,?)
sel_in = Input(shape=(None,), dtype='int32')   # 被select的列  shape=(?,?)
conn_in = Input(shape=(1,), dtype='int32') # 连接类型  shape=(?,1)
#csel_in = Input(shape=(None,), dtype='int32')  # 条件中的列 shape=(?,?)
csel_in = Input(shape=(None,),dtype='int32')  # 条件中的列 shape=(?,?)
cop_in = Input(shape=(None,), dtype='int32') # 条件中的运算符（同时也是值的标记）shape=(?,?)

x1, x2, x3, xm, h, hm, sel, conn, csel, cop = (x1_in, x2_in, x3_in, xm_in, h_in, hm_in, sel_in, conn_in, csel_in, cop_in)

# 第一个部分
hm = Lambda(lambda x: K.expand_dims(x, 1))(hm) # header的mask.shape=(None, 1, h_len)  #列名mask shape=(?,1,?)
x = bert_model([x1_in, x2_in])  # bert的输入  shape=(?,?,768)
# print(x)

# 第一种方法　直接concat
# m = Lambda(lambda x: K.expand_dims(x, 1))(x[:, 0])  # (?,1,768)
# print(m)
n = Lambda(lambda x: K.expand_dims(x, 1))(x3)  # (?, 1, 768)
# print(n)
'''

x_new = Lambda(lambda x: K.concatenate(x[0], x[1]))([m,n])
print(x_new)
'''

# 第二种方法　直接concat
# x_new = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=-2))([m, n])  # (?, 2, 768)
x_new = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))([x, n]) # (?, ?+1,768)
x_new = Lambda(lambda x: x[:,1:])(x_new)
# print(x_new)




# 第二个部分
x4conn = Lambda(lambda x: x[:, 0])(x)  # 对于bert的输入的处理 shape=(?,768)
pconn = Dense(num_cond_conn_op, activation='softmax')(x4conn)  # 连接类型的概率 conn shape=(?,3) {and, or, ''}


# 第三个部分
x4h = Lambda(seq_gather)([x, h])  # 列名所在位置  shape=(?,?,768)
psel = Dense(num_agg, activation='softmax')(x4h) # 被选择列的概率 shape=(?,?,7) {}

# 第四个部分
pcop = Dense(num_op, activation='softmax')(x_new)  # 被选择的条件中的运算符（同时也是值的标记）的概率 shape=(?,?,5)
# print(pcop)


x = Lambda(lambda x: K.expand_dims(x, 2))(x)  # shape=(?,?,1,768)
x4h = Lambda(lambda x: K.expand_dims(x, 1))(x4h)  # 列名所在位置 shape=(?,1,?,768)

pcsel_1 = Dense(256)(x)  # 选择条件中的列的概率 shape=(?,?,1,256)
pcsel_2 = Dense(256)(x4h)  # 选择条件中的列的概率 shape=(?,1,?,256)
pcsel = Lambda(lambda x: x[0] + x[1])([pcsel_1, pcsel_2]) # 选择条件中的列的概率　（?,?,?,256）
pcsel = Activation('tanh')(pcsel)
pcsel = Dense(1)(pcsel) # (?,?,?,1) // (?,?,?)
pcsel = Lambda(lambda x: x[0][..., 0] - (1 - x[1]) * 1e10)([pcsel, hm])  # mask 机制
pcsel = Activation('softmax')(pcsel)  #shape=(?,?,?)



# 第一种模型 加载模型进行训练
model = Model([x1_in, x2_in, x3_in, h_in, hm_in], [psel, pconn, pcop, pcsel])

# 第二种模型 训练过程
train_model = Model(
    [x1_in, x2_in, x3_in, xm_in, h_in, hm_in, sel_in, conn_in, csel_in, cop_in],
    [psel, pconn, pcop, pcsel]
)
# 三个mask 文本mask，headermask，
xm = xm  # question的mask.shape=(None, x_len)  # 输入序列的mask
hm = hm[:, 0] # header的mask.shape=(None, h_len)
cm = K.cast(K.not_equal(cop, num_op - 1), 'float32') # conds的mask.shape=(None, x_len)

# 主要是计算关于选择列，条件中欧的列，链接类型，条件中的操作符号损失函数
psel_loss = K.sparse_categorical_crossentropy(sel_in, psel)  # sel_in=(?,?), psel=(?,?,7)
psel_loss = K.sum(psel_loss * hm) / K.sum(hm)  #

pconn_loss = K.sparse_categorical_crossentropy(conn_in, pconn)  #conn_in=(?,1), pconn=(?,3)
pconn_loss = K.mean(pconn_loss)  #

pcop_loss = K.sparse_categorical_crossentropy(cop_in, pcop) # (?,?)
# print('pcop_loss is', pcop_loss)
pcop_loss = K.sum(pcop_loss * xm) / K.sum(xm)


pcsel_loss = K.sparse_categorical_crossentropy(csel, pcsel) # 主要问题 条件中选择的列  csel_in=(?,?),pcsel=(?,?,?)
pcsel_loss = K.sum(pcsel_loss * xm * cm) / K.sum(xm * cm)
loss = psel_loss + pconn_loss + pcop_loss + pcsel_loss

train_model.add_loss(loss)
train_model.compile(optimizer=Adam(learning_rate))
train_model.summary()


def nl2sql(question, table):
    """
    输入question和headers，转SQL
    """
    x1, x2 = tokenizer.encode(question)
    x3 = achieve_feature1(question, table['headers-types'])[0].detach().numpy().tolist()  # [1,768]
    # print('x3 is ',x3)
    h = []
    for i in table['headers']:
        _x1, _x2 = tokenizer.encode(i)
        h.append(len(x1))
        x1.extend(_x1)
        x2.extend(_x2)
    hm = [1] * len(h)
    psel, pconn, pcop, pcsel = model.predict([np.array([x1]), np.array([x2]), np.array([x3]), np.array([h]), np.array([hm])])
    R = {'agg': [], 'sel': []}
    for i, j in enumerate(psel[0].argmax(1)):
        if j != num_agg - 1: # num_agg-1类是不被select的意思
            R['sel'].append(i)
            R['agg'].append(j)
    conds = []
    v_op = -1
    for i, j in enumerate(pcop[0, :len(question)+1].argmax(1)): # (?,?,5)  # (csel, v_op, v_strkeras
        # 这里结合标注和分类来预测条件
        if j != num_op - 1:
            if v_op != j:
                if v_op != -1:
                    v_end = v_start + len(v_str)
                    csel = pcsel[0][v_start: v_end].mean(0).argmax()
                    conds.append((csel, v_op, v_str))
                v_start = i
                v_op = j
                v_str = question[i - 1]
            else:
                v_str += question[i - 1]
        elif v_op != -1:
            v_end = v_start + len(v_str)
            csel = pcsel[0][v_start: v_end].mean(0).argmax()
            conds.append((csel, v_op, v_str))
            v_op = -1
    R['conds'] = set()
    for i, j, k in conds:
        if re.findall('[^\d\.]', k):
            j = 2 # 非数字只能用等号
        if j == 2:
            if k not in table['all_values']:
                # 等号的值必须在table出现过，否则找一个最相近的
                k = most_similar(k, list(table['all_values']))
            h = table['headers'][i]
            # 然后检查值对应的列是否正确，如果不正确，直接修正列名
            if k not in table['content'][h]:
                for r, v in table['content'].items():
                    if k in v:
                        i = table['header2id'][r]
                        break
        R['conds'].add((i, j, k))
    R['conds'] = list(R['conds'])
    if len(R['conds']) <= 1: # 条件数少于等于1时，条件连接符直接为0 “”
        R['cond_conn_op'] = 0
    else:
        R['cond_conn_op'] = 1 + pconn[0, 1:].argmax() # 不能是0
    return R


def is_equal_all(R1, R2):
    """
    判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op']) &\
    (set(zip(R1['sel'], R1['agg'])) == set(zip(R2['sel'], R2['agg']))) &\
    (set([tuple(i) for i in R1['conds']]) == set([tuple(i) for i in R2['conds']]))

def is_equal_sel(R1, R2):# 计算sel　agg 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (set(zip(R1['sel'])) == set(zip(R2['sel'])))

def is_equal_agg(R1, R2):# 计算sel　agg 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (set(zip( R1['agg'])) == set(zip(R2['agg'])))

def is_equal_conn(R1, R2):# 计算cond_conn_p 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op'])

def is_equal_where_sel(R1, R2):# 计算cond_ 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (set([tuple(i)[0] for i in R1['conds']]) == set([tuple(i)[0] for i in R2['conds']]))

def is_equal_where_op(R1, R2):# 计算cond_ 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (set([tuple(i)[1] for i in R1['conds']]) == set([tuple(i)[1] for i in R2['conds']]))

def is_equal_where_value(R1, R2):# 计算cond_ 匹配度
    """判断两个SQL字典是否全匹配
    """
    return (set([tuple(i)[2] for i in R1['conds']]) == set([tuple(i)[2] for i in R2['conds']]))


def evaluate(data, tables): # valid_data, valid_tables
    right_all = 0.
    right_sel = 0.
    right_agg = 0.
    right_conn = 0.
    right_where_sel = 0.
    right_where_op = 0.
    right_where_value = 0.
    pbar = tqdm()
    F = open('evaluate_pred.json', 'w')
    for i, d in enumerate(data):
        question = d['question']
        # print(question)
        table = tables[d['table_id']]
        R = nl2sql(question, table) # 调用nl2SQL的方法
        right_all += float(is_equal_all(R, d['sql']))
        right_sel += (is_equal_sel(R, d['sql'])) # len(nl2sql(question, table))
        right_agg += (is_equal_agg(R, d['sql']))
        right_conn += (is_equal_conn(R, d['sql']))
        right_where_sel += (is_equal_where_sel(R, d['sql']))
        right_where_op += (is_equal_where_op(R, d['sql']))
        right_where_value += (is_equal_where_value(R, d['sql']))
        pbar.update(1)
        pbar.set_description('< acc: %.5f >' % (right_all / (i + 1)))
        d['sql_pred'] = R
        s = json.dumps(d, ensure_ascii=False, indent=4, cls=NpEncoder)
        F.write(str(s.encode('utf-8') + b'\n'))
    F.close()
    pbar.close()
    return right_all / len(data), right_sel/len(data), right_agg/len(data), right_conn/len(data), right_where_sel/len(data), right_where_op/len(data), right_where_value/len(data)


def test(data, tables, outfile='result.json'):
    test_right_all = 0.
    test_right_sel = 0.
    test_right_agg = 0.
    test_right_conn = 0.
    test_right_where_col = 0.
    test_right_where_op = 0.
    test_right_where_value = 0.
    pbar = tqdm()
    F = open(outfile, 'w')
    for i, d in enumerate(data):
        question = d['question']
        table = tables[d['table_id']]
        R = nl2sql(question, table)
        test_right_all += float(is_equal_all(R, d['sql']))
        test_right_sel += (is_equal_sel(R, d['sql']))  # len(nl2sql(question, table))
        test_right_agg += (is_equal_agg(R, d['sql']))
        test_right_conn += (is_equal_conn(R, d['sql']))
        test_right_where_col += (is_equal_where_sel(R, d['sql']))
        test_right_where_op += (is_equal_where_op(R, d['sql']))
        test_right_where_value += (is_equal_where_value(R, d['sql']))
        pbar.update(1)
        s = json.dumps(R, ensure_ascii=False, cls=NpEncoder)
        F.write(str(s.encode('utf-8') + b'\n'))
    F.close()
    pbar.close()
    print('acc: %.5f, sel_acc: %.5f, sel_agg: %.5f, conn_acc: %.5f,where_col_acc: %.5f,where_op_acc: %.5f,where_value_acc: %.5f\n' % (test_right_all, test_right_sel, test_right_agg, test_right_conn, test_right_where_col, test_right_where_op,test_right_where_value))





dict_acc_all = []
dict_best_acc = []
dict_sel_acc = []
dict_agg_acc = []
dict_conn_acc = []
dict_where_sel_acc = []
dict_where_op_acc = []
dict_where_value_acc = []
class Evaluate(Callback):
    def __init__(self):
        self.accs = []
        self.best = 0.
        self.passed = 0
        self.stage = 0
    def on_batch_begin(self, batch, logs=None):
        """第一个epoch用来warmup，第二个epoch把学习率降到最低
        """
        if self.passed < self.params['steps']:
            lr = (self.passed + 1.) / self.params['steps'] * learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
        elif self.params['steps'] <= self.passed < self.params['steps'] * 2:
            lr = (2 - (self.passed + 1.) / self.params['steps']) * (learning_rate - min_learning_rate)
            lr += min_learning_rate
            K.set_value(self.model.optimizer.lr, lr)
            self.passed += 1
    def on_epoch_end(self, epoch, logs=None):
        acc_all, sel_acc, agg_acc, conn_acc, where_sel_acc, where_op_acc, where_value_acc = self.evaluate()
        self.accs.append(acc_all)
        if acc_all > self.best:
            self.best = acc_all
            train_model.save_weights('model/best_bert_model.weights') ########需要修改#########
        print('acc: %.5f, best acc: %.5f, sel_acc: %.5f, agg_acc: %.5f, conn_acc: %.5f,where_sel_acc: %.5f,where_op_acc: %.5f,where_value_acc: %.5f\n' % (acc_all, self.best, sel_acc,agg_acc, conn_acc, where_sel_acc, where_op_acc, where_value_acc))
        dict_acc_all.append(acc_all)
        dict_best_acc.append(self.best)
        dict_sel_acc.append(sel_acc)
        dict_agg_acc.append(agg_acc)
        dict_conn_acc.append(conn_acc)
        dict_where_sel_acc.append(where_sel_acc)
        dict_where_op_acc.append(where_op_acc)
        dict_where_value_acc.append(where_value_acc)
    def evaluate(self):
        return evaluate(valid_data, valid_tables) # 验证集部分
        # return evaluate(test_data,test_tables) # 测试集部分


train_D = data_generator(train_data, train_tables, achieve_feature=achieve_feature(train_data,train_tables))
# print(train_D)
dev_D = data_generator(valid_data, valid_tables, achieve_feature=achieve_feature(valid_data,valid_tables))
# print(dev_D)


evaluator = Evaluate()

if __name__ == '__main__':
    H = train_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        validation_data=dev_D.__iter__(),
        validation_steps=len(dev_D),
        epochs=100,  # 100
        callbacks=[evaluator]
    )
    print(H.history)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    epochs = len(H.history['loss'])
    plt.title('Loss function')
    plt.plot(range(epochs), H.history['loss'], label='loss')
    plt.plot(range(epochs), H.history['val_loss'], label='val_loss')
    plt.legend()



    plt.subplot(1, 2, 2)
    epochs = len(H.history['loss'])
    plt.title('acc & sel_acc & agg _acc & conn_acc & where_sel_acc & where_op_acc & where_value_acc')
    plt.plot(range(epochs), dict_acc_all, label='val_acc')
    plt.plot(range(epochs), dict_sel_acc, label='val_sel_acc')
    plt.plot(range(epochs), dict_agg_acc, label='val_sel_acc')
    plt.plot(range(epochs), dict_conn_acc, label='val_conn_acc')
    plt.plot(range(epochs), dict_where_sel_acc, label='val_where_sel_acc')
    plt.plot(range(epochs), dict_where_op_acc, label='val_where_op_acc')
    plt.plot(range(epochs), dict_where_sel_acc, label='val_where_sel_acc')
    plt.legend()
    plt.savefig("figure/nl2sql_bert.png")  ########需要修改#########

    print("acc_all的最小值是%.5f,acc_all的最大值是%.5f"% (min(dict_acc_all), max(dict_acc_all)))
    print("acc_sel的最小值是%.5f,acc_sel的最大值是%.5f" % (min(dict_sel_acc), max(dict_sel_acc)))
    print("acc_agg的最小值是%.5f,acc_agg的最大值是%.5f" % (min(dict_agg_acc), max(dict_agg_acc)))
    print("acc_conn的最小值是%.5f,acc_conn的最大值是%.5f" % (min(dict_conn_acc), max(dict_conn_acc)))
    print("acc_where_sel的最小值是%.5f,acc_where_sel的最大值是%.5f" % (min(dict_where_sel_acc), max(dict_where_sel_acc)))
    print("acc_where_op的最小值是%.5f,acc_where_op的最大值是%.5f" % (min(dict_where_op_acc), max(dict_where_op_acc)))
    print("acc_where_value的最小值是%.5f,acc_where_value的最大值是%.5f" % (min(dict_where_value_acc), max(dict_where_value_acc)))

    #test(test_data, test_tables)  # 测试部分


else:
    ########需要修改#########
    train_model.load_weights('model/best_bert_model.weights')








