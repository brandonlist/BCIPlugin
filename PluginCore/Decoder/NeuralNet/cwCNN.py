"""
EEG-based Prediction of Driver’s Cognitive Performance
by Deep Convolutional Neural Network

channel-wise CNN

example:
cwCNN = ChannelWiseCNN(n_chan=64,time_step=320,n_classes=4,n_filter=4,kernel_size=10,drop_p0=0.7)
x = torch.randn((10,1,64,320))
ans = cwCNN(x)
"""
from torch import nn
import torch
import torch.nn.functional as F

class ChannelWiseCNN(nn.Module):
    def __init__(self,n_chan,time_step,n_classes,n_filter,kernel_size,drop_p0):
        super(ChannelWiseCNN, self).__init__()
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.drop_p0 = drop_p0

        self.conv = nn.Conv2d(in_channels=1,out_channels=n_filter,kernel_size=(1,kernel_size))
        self.dropout_0 = nn.Dropout(self.drop_p0)

        self.clf = nn.Linear(in_features=n_filter*n_chan*(time_step-(kernel_size-1)),out_features=n_classes)

    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        feature = self.conv(x)
        feature = self.dropout_0(feature)

        feature = feature.view(feature.shape[0],-1)
        pred = self.clf(feature)
        pred = F.log_softmax(pred)
        return pred


