"""
Learning joint space–time–frequency features for EEG decoding on
small labeled data (using morlet kernel)

CNN, wavelet-kernel:
example:
mCNN = WaSFCNNMorlet(n_chan=64,time_step=320,n_classes=4,n_filter_time=20,n_filter_spat=40,fs=100)
x = torch.randn((10,1,64,320))
ans = mCNN(x)
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MorletConv2d(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0, sample_rate=250,device='cuda'):
        super(MorletConv2d, self).__init__()

        self.device = device
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.sample_rate = sample_rate


        self.a = nn.Parameter(torch.rand((self.channel_out, 1))*(10-1)+1, requires_grad=True)
        self.b = nn.Parameter(torch.rand((self.channel_out, 1))*(40-3)+3, requires_grad=True)

        if self.device=='cuda':
            self.a.cuda()
            self.b.cuda()

    def forward(self, x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        window_len_point = self.kernel_size[0] * self.kernel_size[1]
        window_len_time = window_len_point / self.sample_rate
        t_tensor = torch.from_numpy(np.linspace(0, window_len_time, window_len_point).astype(np.float32)-(window_len_time / 2))
        if self.device=='cuda':
            t_tensor = t_tensor.cuda()
        kernel = torch.exp(-.5 * (torch.pow(self.a.mm(t_tensor.reshape(1, self.kernel_size[0]*self.kernel_size[1])), 2))) * (torch.cos(self.b.mm(2.0 * 3.1415926 * t_tensor.reshape(1, self.kernel_size[0]*self.kernel_size[1]))))
        kernel_r = torch.reshape(kernel, (self.channel_out, self.channel_in, self.kernel_size[0], self.kernel_size[1]))
        out = F.conv2d(x, kernel_r, stride=self.stride, padding=self.padding)

        out = F.log_softmax(out,dim=-1)
        return out

class WaSFCNNMorlet(nn.Module):
    def __init__(self,n_chan,time_step,n_classes,n_filter_time,n_filter_spat,fs,device='cuda'):
        super(WaSFCNNMorlet, self).__init__()
        self.device = device

        self.n_filter_time = n_filter_time
        self.n_filter_spat = n_filter_spat

        self.fs = fs

        self.time_conv = MorletConv2d(channel_in=1, channel_out=self.n_filter_time, kernel_size=(1, 25), sample_rate=self.fs)
        self.spat_conv = nn.Conv2d(in_channels=self.n_filter_time, out_channels=n_filter_spat, kernel_size=(n_chan, 1))
        self.pooling = nn.AvgPool2d((1, 26), stride=(1, 13))
        self.dropout = nn.Dropout(p=0.6)
        self.non_linear1 = nn.ReLU()
        self.non_linear2 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(self.n_filter_time)
        self.bn2 = nn.BatchNorm2d(self.n_filter_spat)
        self.csp_conv = nn.Conv2d(in_channels=self.n_filter_time,out_channels=self.n_filter_time,kernel_size=(1,1))
        self.spat_nin = nn.Conv2d(in_channels=self.n_filter_spat, out_channels=self.n_filter_spat, kernel_size=(1, 1))

        tmp = torch.randn((1,1,n_chan,time_step))
        feature = self.time_conv(tmp)
        feature = self.csp_conv(feature)
        feature = self.spat_conv(feature)
        feature = self.spat_nin(feature)
        out = self.pooling(feature)

        self.clf = nn.Linear(in_features=out.shape[1]*out.shape[3],out_features=n_classes)

        if self.device=='cuda':
            self.time_conv.cuda()

    def forward(self, x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        x = self.time_conv(x)
        x = self.csp_conv(x)
        x = self.bn1(x)
        x = self.non_linear1(x)

        x = self.spat_conv(x)
        x = self.spat_nin(x)
        x = self.bn2(x)
        x = self.non_linear2(x)

        x = self.pooling(x)
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)

        x = self.clf(x)
        x = F.log_softmax(x, dim=1)
        return x


