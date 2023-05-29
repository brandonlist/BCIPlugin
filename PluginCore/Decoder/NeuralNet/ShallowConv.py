from torch import nn
from ..braindecode.core.modules import Expression, Ensure4d_ext,safe_log, square, Transaction_ext
from torch.nn import init
import torch

class ExtractorMarkI(nn.Sequential):
    def __init__(self,n_chan,n_filter_time=40,kernel_size_time=25,kernel_size_pool=75,stride=15,n_filter_spat=40,kernel_size_spat=5,cuda=False):
        super(ExtractorMarkI, self).__init__()
        self.n_filter_time = n_filter_time
        self.n_filter_spat = n_filter_spat

        self.add_module("ensuredims", Ensure4d_ext())
        self.add_module('conv_time', nn.Conv2d(1, self.n_filter_time, kernel_size=(1, kernel_size_time), stride=(1, 1)))
        self.add_module('conv_spat', nn.Conv2d(self.n_filter_time,self.n_filter_spat,kernel_size=(kernel_size_spat,1),stride=(1,1),bias=False),)
        self.add_module('bn', nn.BatchNorm2d(self.n_filter_spat, momentum=0.1, affine=True))
        self.add_module('non_linear', Expression(square))
        self.add_module('pool_time', nn.AvgPool2d(kernel_size=(1, kernel_size_pool), stride=(1, stride)))
        self.add_module('pool_linear', Expression(safe_log))
        self.add_module('drop_out', nn.Dropout(0.5))
        self.add_module('out', Transaction_ext())

        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)
        init.xavier_uniform_(self.conv_spat.weight, gain=1)

        if cuda:
            self.cuda()

class Replicator(nn.Module):
    def __init__(self,n_chan, time_steps, n_classes, n_filter_time=40):
        kernel_size_time = 25
        kernel_size_pool = 75
        stride = 15
        super(Replicator, self).__init__()
        self.extractor = ExtractorMarkI(n_chan=n_chan,n_filter_time=n_filter_time,kernel_size_time=kernel_size_time,
                                   kernel_size_pool=kernel_size_pool,stride=stride)

        tmp = torch.randn(1, n_chan, time_steps)
        out = self.extractor(tmp)

        self.clf = nn.Linear(in_features=out.shape[2]*out.shape[1],out_features=n_classes)
        init.xavier_uniform_(self.clf.weight, gain=1)
        init.constant_(self.clf.bias, 0)

    def forward(self,x):
        feature = self.extractor(x)
        feature = torch.reshape(feature,[x.shape[0],-1])

        logit = self.clf(feature)
        return logit

    def transform(self,x):
        feature = self.extractor(x)
        return feature

class Reconstructor(nn.Module):
    def __init__(self, n_chan, time_steps, extractor, cuda):
        super(Reconstructor, self).__init__()
        self.extractor = extractor

        tmp = torch.randn((1, n_chan, time_steps))
        if cuda:
            tmp = tmp.cuda()
            self.extractor.cuda()
        out = self.extractor(tmp)
        dim_ch = out.shape[1];dim_ti = out.shape[2]

        kernel_size = 5
        self.channel_decoder = nn.ConvTranspose1d(in_channels=dim_ch, out_channels=n_chan, kernel_size=kernel_size)
        self.time_decoder = nn.Linear(in_features=dim_ti + kernel_size - 1, out_features=time_steps)

        del self.extractor

        if cuda:
            self.cuda()

    def forward(self, x):
        recon = self.channel_decoder(x)
        recon = self.time_decoder(recon)
        return recon
