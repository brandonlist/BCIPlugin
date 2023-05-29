"""
Multimodal Neural Network for Rapid Serial Visual
Presentation Brain Computer Interface

CNN, input being square
example:
rc = RSVPConv(n_chan=64,time_step=320,n_classes=4)
x = torch.randn((10,1,64,320))
ans = rc(x)

"""
from torch import nn
import torch
import torch.nn.functional as F

class RSVPConv(nn.Module):
    def __init__(self,n_chan,time_step,n_classes):
        super(RSVPConv, self).__init__()

        self.conv_0 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=(n_chan,1)),
            nn.ELU(),
        )

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=128,kernel_size=(1,9)),
            nn.MaxPool2d(kernel_size=(1,4),stride=(1,4)),
            nn.ELU(),
            nn.Dropout()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 9)),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.ELU(),
            nn.Dropout()
        )

        tmp = torch.randn((1,1,n_chan,time_step))
        out = self.conv_0(tmp)
        out = self.conv_1(out)
        out = self.conv_2(out)

        self.clf = nn.Sequential(
            nn.Linear(in_features=out.shape[1]*out.shape[3],out_features=n_classes)
        )

    def forward(self,x):
        if x.ndim==3:
            x = torch.unsqueeze(x,dim=1)

        feature = self.conv_0(x)
        feature = self.conv_1(feature)
        feature = self.conv_2(feature)

        feature = feature.view(feature.shape[0],-1)
        pred = self.clf(feature)
        pred = F.log_softmax(pred,dim=-1)
        return pred

