#@Author:   Casserole fish
#@Time:    2021/3/1 19:07

import numpy as np
import torch
import torch.utils.data as Data

#文本预处理
sentences = ["jack like dog", "jack like cat", "jack like animal",
  "dog cat animal", "banana apple cat dog like", "dog fish milk like",
  "dog cat animal like", "jack like apple", "apple like", "jack like banana",
  "apple banana jack movie book music like", "cat dog hate", "cat dog like"]

#先将列表中句子连接起来，然后按照空格切分
word_sequence = " ".join(sentences).split()# ['jack', 'like', 'dog', 'jack', 'like', 'cat', 'animal',...]
vocab = list(set(word_sequence))
print(len(vocab))
word2idx={w:i for i,w in enumerate(vocab)}

#数据预处理
#构造skip_grams 列表,window_size 为设置的skip_grams的窗口尺寸
def get_skip_grams(window_size):
    skip_grams=[]
    for idx in range(window_size,len(word_sequence)-window_size):
        center=word2idx[word_sequence[idx]] #center word
        contex_idx=list(range(idx-window_size,idx)) + list(range(idx+1,idx+window_size+1)) #context word
        contex=[word2idx[word_sequence[idx]] for idx in contex_idx]

        for w in contex:
            skip_grams.append([center,w])

    return skip_grams

#构造输入输出数据
def make_data_skip(skip_grams,voc_size):
    input_data=[]
    output_data=[]

    for i in range(len(skip_grams)):
        #np.eye()[]可以将数组转换成one-hot编码形式
        input_data.append(np.eye(voc_size)[skip_grams[i][0]])
        output_data.append(skip_grams[i][1])

    return input_data,output_data

#加载数据
def load_data_skip(skip_grams,voc_size,batch_size=10):
    input_data,output_data = make_data_skip(skip_grams,voc_size)
    input_data,output_data = torch.Tensor(input_data),torch.Tensor(output_data)
    dataset=Data.TensorDataset(input_data,output_data)
    data = Data.DataLoader(dataset,batch_size,True)
    return data


