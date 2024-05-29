from torch import nn
import torch
import torch.nn.functional as F
from loguru import logger

from torch import nn
from collections import OrderedDict


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,activation,stride=1,padding = 1):
        super(Bottleneck, self).__init__()
        self.stride=stride
        self.conv1 = nn.Conv1d(in_channel,in_channel*expansion,kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channel*expansion,in_channel*expansion,kernel_size = 3, groups = in_channel*expansion,
                               padding=padding,stride = stride)
        self.conv3 = nn.Conv1d(in_channel*expansion,out_channel,kernel_size = 1, stride =1)
        self.b0 = nn.BatchNorm1d(in_channel*expansion)
        self.b1 =  nn.BatchNorm1d(in_channel*expansion)
        self.d = nn.Dropout()
        self.act = activation()
    def forward(self,x):
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x+y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y

from torch import nn
from collections import OrderedDict

class MBConv(nn.Module):
    def __init__(self,in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
        super(MBConv, self).__init__()
        self.stack = OrderedDict()
        for i in range(0,layers-1):
            self.stack['s'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
            #self.stack['a'+str(i)] = activation()
        self.stack['s'+str(layers+1)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
        # self.stack['a'+str(layers+1)] = activation()
        self.stack = nn.Sequential(self.stack)
        
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self,x):
        x = self.stack(x)
        return self.bn(x)


"""def MBConv(in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
    stack = OrderedDict()
    for i in range(0,layers-1):
        stack['b'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
    stack['b'+str(layers)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
    return nn.Sequential(stack)"""


class EffNet(nn.Module):
    
    def __init__(
            self, 
            in_channel, 
            num_additional_features = 0, 
            depth = [1,2,2,3,3,3,3], 
            channels = [32,16,24,40,80,112,192,320,1280],
            dilation = 1,
            stride = 2,
            expansion = 6):
        super(EffNet, self).__init__()
        logger.info(f"depth: {depth}")
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) # 
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv
        
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, 1)
        
        
    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 12 x 2500
        x = self.b0(self.stage1(x))
        # N x 32 x 1250
        x = self.stage2(x)
        # N x 16 x 625
        x = self.stage3(x)
        # N x 24 x 313
        x = self.Pool(x)
        # N x 24 x 313
        
        x = self.stage4(x)
        # N x 40 x 157
        x = self.stage5(x)
        # N x 80 x 79
        x = self.stage6(x)
        # N x 112 x 40
        x = self.Pool(x)
        # N x 192 x 20
        
        x = self.stage7(x)
        # N x 320 x 10
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 10
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        x = self.fc(x)
        # N x 1
        return x


class EffNetSupervised(nn.Module):
    
    def __init__(
            self, 
            in_channel, 
            num_classes = 5,
            num_additional_features = 0, 
            depth = [1,2,2,3,3,3,3], 
            channels = [32,16,24,40,80,112,192,320,1280],
            dilation = 1,
            stride = 2,
            expansion = 6):
        super(EffNetSupervised, self).__init__()
        logger.info(f"depth: {depth}")
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) # 
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv
        
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, num_classes)
        
    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 12 x 2500
        x = self.b0(self.stage1(x))
        # N x 32 x 1250
        x = self.stage2(x)
        # N x 16 x 625
        x = self.stage3(x)
        # N x 24 x 313
        x = self.Pool(x)
        # N x 24 x 313
        
        x = self.stage4(x)
        # N x 40 x 157
        x = self.stage5(x)
        # N x 80 x 79
        x = self.stage6(x)
        # N x 112 x 40
        x = self.Pool(x)
        # N x 192 x 20
        
        x = self.stage7(x)
        # N x 320 x 10
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 10
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        return x
