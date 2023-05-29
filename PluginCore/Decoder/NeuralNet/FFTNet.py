"""
Convolutional Neural Network with embedded Fourier Transform for EEG
classification

CNN + FT, used for SSVEP
example:
x = torch.randn((10,1,32,320))
model = WaveFFTNet(in_chan=32,time_step=320,n_classes=4)
ans = model(x)
"""
from torch import nn
import torch
import torch.nn.functional as F

class WaveFFTNet(nn.Module):
    def __init__(self,in_chan,time_step,n_classes,n_freq=64):
        super(WaveFFTNet, self).__init__()
        self.n_feature = 100

        self.wave_conv = nn.Conv2d(in_channels=1,out_channels=self.n_feature,kernel_size=(in_chan,8))
        self.n_freq = n_freq
        self.fc0 = nn.Linear(in_features=self.n_feature*self.n_freq,out_features=self.n_feature)
        self.clf = nn.Linear(in_features=self.n_feature,out_features=n_classes)

    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        feature = self.wave_conv(x)
        feature = torch.fft.fft(feature,n=self.n_freq)
        feature = torch.abs(feature)
        feature = feature.view(feature.shape[0],-1)
        pred = self.fc0(feature)
        pred = F.elu(pred)
        pred = self.clf(pred)
        pred = F.softmax(pred)

        pred = F.log_softmax(pred,dim=-1)
        return pred


