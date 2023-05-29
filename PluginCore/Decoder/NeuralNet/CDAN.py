import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init

class GRL(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input
    @staticmethod
    def backward(ctx,grad_output):
        grad_input = grad_output.neg()
        return grad_input

class CDANExtractor(nn.Module):
    def __init__(self, n_chan, time_steps):
        # Channel-Projection Layer, temporal Convolution Layer and Spatial Convolution Layer
        super(CDANExtractor, self).__init__()
        n_project_feature = 35
        dim_feature = 4

        self.ChannelProjectionLayer = nn.Sequential(
            nn.Conv2d(in_channels=n_chan, out_channels=n_project_feature, kernel_size=(1, 1)),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=n_project_feature),
        )

        self.TemporalConvLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 25)),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=1),
        )

        self.SpatialConvLayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=dim_feature, kernel_size=(n_project_feature,1)),
            nn.Dropout(),
            nn.BatchNorm2d(num_features=dim_feature),
            nn.MaxPool2d(kernel_size=(1,3)),
            nn.ELU()
        )

        self.DensePreConv = nn.Conv2d(in_channels=dim_feature,out_channels=dim_feature,kernel_size=(1,1))
        self.DensePostConv = nn.Sequential(
            nn.Conv2d(in_channels=dim_feature,out_channels=dim_feature,kernel_size=(1,11)),
            nn.MaxPool2d(kernel_size=(1,3))
        )

    def ForwardFeature(self, x):
        b_s, n_chan, time_steps = x.shape[0], x.shape[1], x.shape[2]
        x = torch.unsqueeze(x, dim=2)
        feature = self.ChannelProjectionLayer(x)
        # [b_z, n_project_feature(35), 1, time_steps-1]
        feature = torch.swapaxes(feature, 1, 2)
        # [b_z, 1, n_project_feature(35), time_steps-1]
        feature = self.TemporalConvLayer(feature)
        # [b_z, 1, n_project_feature(35), ?]
        feature = self.SpatialConvLayer(feature)
        # [b_z, dim_feature, 1, ?]
        feature = feature.clone() + self.DensePreConv(feature)
        feature = self.DensePostConv(feature)

        feature = torch.reshape(feature,(b_s,-1))
        return feature

    def forward(self, x):
        feature = self.ForwardFeature(x)
        return feature

class CDANClassifier(nn.Module):
    def __init__(self, dim_feature, n_class):
        super(CDANClassifier, self).__init__()
        self.clf = nn.Linear(in_features=dim_feature,out_features=n_class)
        init.xavier_uniform_(self.clf.weight, gain=1)
        init.constant_(self.clf.bias, 0)

    def forward(self, x):
        logits = self.clf(x)
        pred = F.softmax(logits,dim=1)
        return pred

class Discriminator(nn.Module):
    def __init__(self, dim_feature):
        super(Discriminator, self).__init__()
        self.grl = GRL()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=dim_feature,out_features=128),
            nn.Linear(in_features=128,out_features=2),
            nn.Sigmoid()
        )
        init.xavier_uniform_(self.discriminator[0].weight, gain=1)
        init.constant_(self.discriminator[0].bias, 0)
        init.xavier_uniform_(self.discriminator[1].weight, gain=1)
        init.constant_(self.discriminator[1].bias, 0)

    def forward(self, x):
        x = self.grl.apply(x)
        pred = self.discriminator(x)
        return pred

    def init_params(self):
        init.xavier_uniform_(self.discriminator[0].weight, gain=1)
        init.constant_(self.discriminator[0].bias, 0)
        init.xavier_uniform_(self.discriminator[1].weight, gain=1)
        init.constant_(self.discriminator[1].bias, 0)

class CDAN(nn.Module):
    def __init__(self, extractor, clf):
        super(CDAN, self).__init__()
        self.extractor = extractor
        self.clf = clf

    def forward(self, x):
        feature = self.extractor(x)
        logits = self.clf(feature)
        return logits

def kronecker_product(mat1, mat2):
    # 在pytorch1.7版本之后，torch.ger就被torch.outer所代替了
    out_mat = torch.ger(mat1.view(-1), mat2.view(-1))
    # 这里的(mat1.size() + mat2.size())表示的是将两个list拼接起来
    out_mat = out_mat.reshape(*(mat1.size() + mat2.size())).permute([0, 2, 1, 3])
    out_mat = out_mat.reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))
    return out_mat
