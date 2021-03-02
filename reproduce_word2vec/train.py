#@Author:   Casserole fish
#@Time:    2021/3/2 16:48

from model import *
from tools import *
import torch
import torch.optim as optim
import scipy
from sklearn.metrics.pairwise import cosine_similarity

WINDOW_SIZE=3 #context window
K = 15 #number of negative samples
EPOCHS = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
BATCH_SIZE= 128
lr = 0.2
device='cpu'
text_path='./data/text8/text8.train.txt'
text,word2idx,idx2word,word_freqs=preprocess(text_path,vocab_size=MAX_VOCAB_SIZE)

SkipGram=SkipGramModel(vocab_size=MAX_VOCAB_SIZE,embedding_size=EMBEDDING_SIZE).to(device)
optimizer=optim.Adam(SkipGram.parameters(),lr=lr)

dataloader = load_data(text=text,word2idx=word2idx,word_freqs=word_freqs,window_size=WINDOW_SIZE,K=K,batch_size=BATCH_SIZE)


def train():
    loss_batch=[]
    for idx,data in enumerate(dataloader):
        print('batch_idx:',idx)
        optimizer.zero_grad()
        input_labels = data[0].long().to(device)
        pos_labels = data[1].long().to(device)
        neg_labels = data[2].long().to(device)

        loss = SkipGram(input_labels,pos_labels,neg_labels).mean()
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())

    return np.mean(loss_batch)

def find_nearest(word):
    index = word2idx[word]
    embedding_weights=SkipGramModel.input_embedding()
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]

for epoch in range(EPOCHS):
    print('epoch:',epoch)
    loss=train()
    print('loss:',loss)


for word in ["two", "america", "computer"]:
    print(word, find_nearest(word))
