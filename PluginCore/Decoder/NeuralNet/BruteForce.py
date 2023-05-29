"""
A Brute-Force CNN Model Selection for Accurate Classification
of Sensorimotor Rhythms in BCIs

CNN model architecture selector

example:
x = torch.randn((10,1,64,320))

for i in range(2,5):
    for j in [9,25,41]:
        print('Model L',i,'-K(3x',j,')')
        model = BruteForce(n_chan=64, time_step=320, depth_fun='default_0', L=i, n_classes=4, kernel_sz=j)
        ans = model(x)

"""
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init


class BruteForceNet(nn.Module):
    def __init__(self,n_chan,time_step,depth_fun,L,n_classes,kernel_sz=9,cuda=True,dropout_p=0.8):
        super(BruteForceNet, self).__init__()

        self.L = L

        if depth_fun=='default_0':
            depth_fun = lambda x: int(np.power(2,2+x))
        elif depth_fun=='default_1':
            depth_fun = lambda x: int(np.power(2,self.L+3-x))

        depths = []
        for i in range(0,self.L):
            j = i + 1
            depths.append(depth_fun(j))
        depths.append(1)
        depths.reverse()

        self.conv_blocks = nn.Sequential()

        for i in range(0,self.L):
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=depths[i],out_channels=depths[i+1],kernel_size=(3,kernel_sz),padding=(1,kernel_sz//2)),
                nn.BatchNorm2d(num_features=depths[i+1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
            )
            self.conv_blocks.add_module('conv'+str(i),conv_block)


        tmp = torch.randn((1,1,n_chan,time_step))
        for i in range(0,self.L):
            tmp = self.conv_blocks[i](tmp)
        dim = tmp.shape[1]*tmp.shape[2]*tmp.shape[3]

        self.dropout = nn.Dropout(dropout_p)
        self.clf = nn.Linear(in_features=dim,out_features=n_classes)

        init.xavier_uniform_(self.clf.weight, gain=1)
        init.constant_(self.clf.bias, 0)

        if cuda:
            self.cuda()


    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        x = self.conv_blocks(x)

        feature = self.dropout(x)
        feature = feature.view(feature.shape[0],-1)

        pred = self.clf(feature)
        pred = F.log_softmax(pred,dim=-1)
        return pred

