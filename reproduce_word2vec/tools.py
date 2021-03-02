#@Author:   Casserole fish
#@Time:    2021/3/2 16:48

import numpy as np
from collections import Counter
import torch.utils.data as tud
import torch
#对文本进行预处理
def preprocess(text_path,vocab_size):
    with open(text_path) as f:
        text = f.read() #读取出文本的内容

    #将文本分割成单词列表
    text = text.lower().split()
    #得到单词字典表，key是单词，value是对应的出现的次数
    #Counter 直接将列表转换为对应key为单词，value为对应次数的字典，
    #most_common(n) top n问题
    vocab_dict=dict(Counter(text).most_common(vocab_size-1))
    #将不常用的单词编码为<UNK>
    vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))

    #word2idx
    word2idx={word:i for i,word in enumerate(vocab_dict.keys())}
    #idx2word
    idx2word={i:word for i ,word in enumerate(vocab_dict.keys())}
    #wordcounts
    word_counts = np.array([count for count in vocab_dict.values()])
    #word freqs
    word_freqs = word_counts/np.sum(word_counts)
    word_freqs = word_freqs**(3/4) #论文中推荐这么做
    return text,word2idx,idx2word,word_freqs

#加载数据
class SkipGramWordEmbeddingDateset(tud.Dataset):
    ''' text: a list of words, all text from the training dataset
        word2idx: the dictionary from word to index
        word_freqs: the frequency of each word
    '''
    def __init__(self,text,word2idx,word_freqs,window_size,K):
        #通过父类初始化这个模型，然后重写这个方法
        super(SkipGramWordEmbeddingDateset,self).__init__()
        #把单词数字化表示
        self.text_encoded = [word2idx.get(word,word2idx['<UNK>']) for word in text]
        #nn.Embedding 需要传入LongTensor 类型
        self.text_encoded = torch.Tensor(self.text_encoded)
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        self.window_size = window_size
        #负采样时 每采样一个正确单词，就采样K个错误单词
        self.K = K
    def __len__(self):
        #返回所有单词总数
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''

        #获得中间词
        center_word=self.text_encoded[idx]
        #取得中间词左右两边词的索引
        pos_indices = list(range(idx-self.window_size,idx))+list(range(idx+1,idx+1+self.window_size))
        #取余，避免索引越界
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        #获得两边的词
        pos_words = self.text_encoded[pos_indices]
        '''
        torch.multinomial(input, num_samples,replacement=False, out=None) → LongTensor
        作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
        输入是一个input张量，一个取样数量，和一个布尔值replacement。
        input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素
        被取干净之前，这个元素是不会被取到的。self.word_freqs数值越大，取样概率越大
        n_samples是每一行的取值次数，该值不能大于每一样的元素数，否则会报错。
        replacement指的是取样时是否是有放回的取样，True是有放回，False无放回
        '''
        #采样neg_words 负采样
        neg_words = torch.multinomial(self.word_freqs,self.K*pos_words.shape[0],True)

        #while 循环保证 neg_words中没有背景词
        #背景词和错误的单词有交集
        while (len(set(list(pos_words.numpy())) & set(list(neg_words.numpy())) )>0):
            # 采样neg_words 负采样
            neg_words = torch.multinomial(self.word_freqs, self.K * pos_words.shape[0], True)

        return center_word,pos_words,neg_words

def load_data(text,word2idx,word_freqs,window_size,K,batch_size=512):
    dataset = SkipGramWordEmbeddingDateset(text,word2idx,word_freqs,window_size,K)
    dataloader = tud.DataLoader(dataset,batch_size,shuffle=True)
    return dataloader
