#@Author:   Casserole fish
#@Time:    2021/3/2 16:47

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(SkipGramModel,self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.in_embed = nn.Embedding(self.vocab_size,self.embedding_size)
        self.out_embed = nn.Embedding(self.vocab_size,self.embedding_size)

    def forward(self,input_labels,pos_labels,neg_labels):
        ''' input_labels: center words, [batch_size]
             pos_labels: positive words, [batch_size, (window_size * 2)]
             neg_labels：negative words, [batch_size, (window_size * 2 * K)]

             return: loss, [batch_size]
         '''
        input_embedding = self.in_embed(input_labels)#[batch_size,embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)# [batch_size, (window * 2 * K), embed_size]
        input_embedding = input_embedding.unsqueeze(2)#[batch_size,embed_size,1]

        pos_dot = torch.bmm(pos_embedding,input_embedding) #[batch_size,(window_size *2),1]
        pos_dot = pos_dot.squeeze(2) #[batch_size,(window *2)]

        neg_dot = torch.bmm(neg_embedding,input_embedding) #[batch_size,window_size*2*K,1]
        neg_dot = neg_dot.squeeze(2) #[batch_size,window*2*K]

        #
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        #log_pos 越大越好，log_neg越小越好
        loss = log_pos - log_neg

        return -loss

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()



