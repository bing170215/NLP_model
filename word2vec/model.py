#@Author:   Casserole fish
#@Time:    2021/3/1 17:21

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np

#model skip_gram
#toy model
class SkipGramModel(nn.Module):
    def __init__(self,voc_size,embedding_size):
        super(SkipGramModel,self).__init__()

        self.W=Parameter(torch.randn(voc_size,embedding_size))
        self.V=Parameter(torch.randn(embedding_size,voc_size))

    def forward(self,X):
        #X:[batch_size,voc_size] one-hot
        #torch.mm only for 2 dim matrix, but torch.matmul can use to any dim
        # 矩阵乘法
        hidden=torch.matmul(X,self.W)
        output=torch.matmul(hidden,self.V)

        return output


