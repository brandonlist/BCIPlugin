"""
DeepSleepNet: a Model for Automatic Sleep Stage
Scoring based on Raw Single-Channel EEG

single channel long data
example:
fs = 250
model = DeepSleepNet(n_chan=1,time_step=fs*45,n_classes=4,fs=fs,lstm_hidden_size=512)
x = torch.randn((10,1,1,fs*45))
ans = model(x)
"""
from torch import nn
import torch
import torch.nn.functional as F

class DeepSleepNet(nn.Module):
    """
    This architecture is used to capture long-range data

    """
    def __init__(self,n_chan,time_step,n_classes,fs,lstm_hidden_size):
        super(DeepSleepNet, self).__init__()

        self.fs = fs
        self.n_filter_0 = 64
        self.n_filter_1 = 32
        self.lstm_hidden_size = lstm_hidden_size

        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.n_filter_0, kernel_size=(1, int(self.fs / 2)),
                      stride=(1, int(self.fs / 16))),
            nn.MaxPool2d(kernel_size=(1,8),stride=(1,8)),
            nn.Dropout(),
            nn.Conv2d(in_channels=self.n_filter_0,out_channels=128,kernel_size=(1,8)),
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(1,8)),
            nn.Conv2d(in_channels=64,out_channels=self.n_filter_1,kernel_size=(1,8)),
            nn.MaxPool2d(kernel_size=(1,8),stride=(1,8))
        )

        self.conv_r = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.n_filter_0, kernel_size=(1,int(self.fs*4)),stride=(1,int(self.fs/2))),
            nn.MaxPool2d(kernel_size=(1,4),stride=(1,2)),
            nn.Dropout(),
            nn.Conv2d(in_channels=self.n_filter_0,out_channels=128,kernel_size=(1,6)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 6)),
            nn.Conv2d(in_channels=64, out_channels=self.n_filter_1, kernel_size=(1, 6)),
            nn.MaxPool2d(kernel_size=(1,2),stride=(1,2))
        )

        tmp = torch.randn((1,1,n_chan,time_step))
        out_0 = self.conv_l(tmp)
        out_1 = self.conv_r(tmp)

        lstm_feature_dim = n_chan*(out_0.shape[3]+out_1.shape[3])
        self.fc_res = nn.Linear(in_features=self.n_filter_1*n_chan*(out_0.shape[3]+out_1.shape[3]),out_features=self.lstm_hidden_size)

        self.drop_out_0 = nn.Dropout()
        self.lstm_0 = nn.LSTM(input_size=lstm_feature_dim,hidden_size=self.lstm_hidden_size,num_layers=1
                                ,batch_first=True,bidirectional=False,dropout=0.5)
        self.drop_out_1 = nn.Dropout()
        self.lstm_1 = nn.LSTM(input_size=self.lstm_hidden_size,hidden_size=self.lstm_hidden_size,num_layers=1
                                ,batch_first=True,bidirectional=False,dropout=0.5)
        self.drop_out_2 = nn.Dropout()

        self.drop_out_3 = nn.Dropout()

        self.clf = nn.Linear(self.lstm_hidden_size,out_features=n_classes)

    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)
        batch_size = x.shape[0]

        feature_l = self.conv_l(x)
        feature_r = self.conv_r(x)

        feature = torch.cat([feature_l,feature_r],dim=3)
        feature = feature.view(batch_size,self.n_filter_1,-1)

        feature = self.drop_out_0(feature)

        h00,c00 = torch.randn(1 , batch_size, self.lstm_hidden_size),torch.randn(1 , batch_size, self.lstm_hidden_size)
        output_0, (hn0, cn0) = self.lstm_0(feature, (h00, c00))

        hn0 = self.drop_out_1(hn0)

        h01,c01 = torch.randn(1 , 1, self.lstm_hidden_size),torch.randn(1 , 1, self.lstm_hidden_size)
        output_1, (hn1, cn1) = self.lstm_1(hn0, (h01, c01))
        hn1 = self.drop_out_2(hn1)

        feature_res = feature.view(batch_size,-1)
        res_f = self.fc_res(feature_res)

        feature = res_f + hn1
        feature = self.drop_out_3(feature)

        pred = self.clf(feature)
        pred = torch.squeeze(pred,dim=0)
        pred = F.log_softmax(pred,dim=-1)
        return pred



