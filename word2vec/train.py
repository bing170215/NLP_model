#@Author:   Casserole fish
#@Time:    2021/3/1 20:42
import torch
import torch.optim as optim
from model import *
from tools import *
import argparse

#定义一些变量
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_num', type=int, default=2000, help='Number of epochs, default=10')
parser.add_argument('--batch_size', type=int, default=8, help='Number of epochs, default=10')
parser.add_argument('--window_size', type=int, default=2, help='window size, default=10')
parser.add_argument('--embedding_size', type=int, default=2, help='embedding size, default=10')
parser.add_argument('--voc_size', type=int, default=13, help='vocabular size, default=10')
parser.add_argument('--device_id', type=int, default=0, help='device id, default=10')
args = parser.parse_args()

#set_gpu_id
#torch.cuda.set_device(args.device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkipGramModel(voc_size=args.voc_size,embedding_size=args.embedding_size).to(device)
#optimizer
optimizer=optim.Adam(model.parameters(),lr=1e-3)

#交叉熵误差
criterion = nn.CrossEntropyLoss().to(device)

#加载数据
skip_grams=get_skip_grams(args.window_size)
data_loader = load_data_skip(skip_grams,voc_size=args.voc_size,batch_size=args.batch_size)
def train():
    loss_batch=[]
    for idx,data in enumerate(data_loader):
        optimizer.zero_grad()
        data_x = data[0].to(device)
        data_y=data[1].to(device)
        pred=model(data_x)
        loss=criterion(pred,data_y.long())

        loss.backward()
        optimizer.step()

        loss_batch.append(loss.item())

    return np.mean(loss_batch)


for epoch in range(args.epoch_num):
    loss=train()
    if epoch%100==0:
        print('epoch:',epoch)
        print('loss:',loss)

