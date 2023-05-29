"""
Improving EEG-Based Motor Imagery Classification via Spatial and
Temporal Recurrent Neural Networks

2 LSTM for temporal and spatial features

example:
x = torch.randn((10,1,64,320))
model = SpatwaveLSTM(in_chan=64,time_step=320,n_classes=4,input_size_wave=32,input_size_spat=32,
                     hidden_size_wave=8,hidden_size_spat=8,dim_final_feature=10)
ans = model(x)
"""
from torch import nn
import torch
import torch.nn.functional as F

class SpatwaveLSTM(nn.Module):
    def __init__(self,in_chan,time_step,n_classes,input_size_wave,input_size_spat,
                 hidden_size_spat,hidden_size_wave,dim_final_feature,device='cpu'):
        super(SpatwaveLSTM, self).__init__()

        self.device = device

        self.input_size_spat = input_size_spat
        self.input_size_wave = input_size_wave
        self.hidden_size_spat = hidden_size_spat
        self.hidden_size_wave = hidden_size_wave
        self.dim_final_feature = dim_final_feature
        self.n_classes = n_classes
        self.input_size_wave = input_size_wave

        self.num_layers_wave = 1
        self.num_layers_spat = 2

        self.fc_wave = nn.Linear(in_features=in_chan,out_features=self.input_size_wave)
        self.fc_final_wave = nn.Linear(in_features=self.hidden_size_wave*self.num_layers_wave,out_features=self.dim_final_feature)
        self.fc_final_spat = nn.Linear(in_features=in_chan*self.hidden_size_spat*self.num_layers_spat,out_features=self.dim_final_feature)

        self.bn_wave = nn.BatchNorm1d(num_features=self.dim_final_feature)
        self.bn_spat = nn.BatchNorm1d(num_features=self.dim_final_feature)

        self.spatLSTM = nn.LSTM(input_size=time_step,hidden_size=self.hidden_size_spat,num_layers=self.num_layers_spat
                                ,batch_first=True,bidirectional=True,dropout=0.5,device=self.device)

        self.waveLSTM = nn.LSTM(input_size=self.input_size_wave,hidden_size=self.hidden_size_wave,num_layers=self.num_layers_wave
                                ,batch_first=True,bidirectional=False,dropout=0.5,device=self.device)

        self.clf = nn.Linear(in_features=dim_final_feature,out_features=self.n_classes)

    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)

        batch_size = x.shape[0]

        if x.ndim == 4:
            assert x.shape[1] == 1
            x = torch.squeeze(x, dim=1)
        x_wave = torch.swapdims(x, 1, 2)

        feature_wave = self.fc_wave(x_wave)
        h0_wave = torch.randn(1, batch_size, self.hidden_size_wave)
        c0_wave = torch.randn(1, batch_size, self.hidden_size_wave)

        if self.device=='cuda':
            h0_wave.cuda()
            c0_wave.cuda()

        output_wave, (hn_wave, cn_wave) = self.waveLSTM(feature_wave, (h0_wave, c0_wave))

        hn_wave = torch.swapdims(hn_wave, 0, 1)
        final_feature_wave = self.fc_final_wave(hn_wave.reshape(batch_size, -1))
        final_feature_wave = self.bn_wave(final_feature_wave)

        h0_spat = torch.randn(2*self.num_layers_spat, batch_size, self.hidden_size_spat)
        c0_spat = torch.randn(2*self.num_layers_spat, batch_size, self.hidden_size_spat)

        if self.device=='cuda':
            h0_spat.cuda()
            c0_spat.cuda()
        output_spat, (hn_spat, cn_spat) = self.spatLSTM(x, (h0_spat, c0_spat))
        final_feature_spat = self.fc_final_spat(output_spat.reshape(batch_size, -1))
        final_feature_spat = self.bn_spat(final_feature_spat)

        final_feature = final_feature_spat + final_feature_wave
        pred = self.clf(final_feature)

        pred = F.log_softmax(pred,dim=-1)
        return pred

