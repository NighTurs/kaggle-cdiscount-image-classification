# https://arxiv.org/pdf/1610.02357.pdf


# "Xception: Deep Learning with Depthwise Separable Convolutions" - Francois Chollet (Google, Inc), CVPR 2017

# separable conv pytorch
#  https://github.com/szagoruyko/pyinn
#  https://github.com/pytorch/pytorch/issues/1708
#  https://discuss.pytorch.org/t/separable-convolutions-in-pytorch/3407/2
#  https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/3

import os
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyinn as P
from pyinn.modules import Conv2dDepthwise

#----- helper functions ------------------------------
BN_EPS = 1e-4  #1e-4  #1e-5

class ConvBn2d(nn.Module):

    def merge_bn(self):
        #raise NotImplementedError
        assert(self.conv.bias==None)
        conv_weight     = self.conv.weight.data
        bn_weight       = self.bn.weight.data
        bn_bias         = self.bn.bias.data
        bn_running_mean = self.bn.running_mean
        bn_running_var  = self.bn.running_var
        bn_eps          = self.bn.eps

        #https://github.com/sanghoon/pva-faster-rcnn/issues/5
        #https://github.com/sanghoon/pva-faster-rcnn/commit/39570aab8c6513f0e76e5ab5dba8dfbf63e9c68c

        N,C,KH,KW = conv_weight.size()
        std = 1/(torch.sqrt(bn_running_var+bn_eps))
        std_bn_weight =(std*bn_weight).repeat(C*KH*KW,1).t().contiguous().view(N,C,KH,KW )
        conv_weight_hat = std_bn_weight*conv_weight
        conv_bias_hat   = (bn_bias - bn_weight*std*bn_running_mean)

        self.bn   = None
        self.conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                              padding=self.conv.padding, stride=self.conv.stride, dilation=self.conv.dilation, groups=self.conv.groups,
                              bias=True)
        self.conv.weight.data = conv_weight_hat #fill in
        self.conv.bias.data   = conv_bias_hat


    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=BN_EPS)

        if is_bn is False:
            self.bn =None

    def forward(self,x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# ----
class SeparableConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, is_bn=True):
        super(SeparableConvBn2d, self).__init__()

        #self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels, bias=False)  #depth_wise
        #self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False) #point_wise

        self.conv1 = Conv2dDepthwise(in_channels,  kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_channels, eps=BN_EPS)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x

#
class SBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels, is_first_relu=True):
        super(SBlock, self).__init__()
        self.is_first_relu=is_first_relu

        self.downsample = ConvBn2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2)
        self.conv1 = SeparableConvBn2d(in_channels,     channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(   channels, out_channels, kernel_size=3, padding=1, stride=1)

    def forward(self,x):
        residual = self.downsample(x)
        if self.is_first_relu:
            x = F.relu(x,inplace=False)
        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, padding=1, stride=2)
        x = x + residual

        return x



class XBlock(nn.Module):

    def __init__(self, in_channels):
        super(XBlock, self).__init__()

        self.conv1 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = SeparableConvBn2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1)

    def forward(self,x):

        residual = x
        x = F.relu(x,inplace=True)
        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.relu(x,inplace=True)
        x = self.conv3(x)
        x = x + residual

        return x



class EBlock(nn.Module):

    def __init__(self, in_channels, channels, out_channels):
        super(EBlock, self).__init__()

        self.conv1 = SeparableConvBn2d(in_channels, channels, kernel_size=3, padding=1, stride=1)
        self.conv2 = SeparableConvBn2d(channels,out_channels, kernel_size=3, padding=1, stride=1)


    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x,inplace=True)
        x = self.conv2(x)
        x = F.relu(x,inplace=True)

        return x


class Xception(nn.Module):

    def load_pretrain_pytorch_file(self,pytorch_file, skip=[]):
        pytorch_state_dict = torch.load(pytorch_file,map_location=lambda storage, loc: storage)
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            #print(key)
            state_dict[key] = pytorch_state_dict[key]
        self.load_state_dict(state_dict)

    #-----------------------------------------------------------------------

    def __init__(self, in_shape=(3,128,128), num_classes=5000 ):
        super(Xception, self).__init__()
        in_channels, height, width = in_shape
        self.num_classes = num_classes

        self.entry0  = nn.Sequential(
            ConvBn2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ConvBn2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.entry1  = SBlock( 64,128,128,is_first_relu=False)
        self.entry2  = SBlock(128,256,256)
        self.entry3  = SBlock(256,728,728)

        self.middle1 = XBlock(728)
        self.middle2 = XBlock(728)
        self.middle3 = XBlock(728)
        self.middle4 = XBlock(728)
        self.middle5 = XBlock(728)
        self.middle6 = XBlock(728)
        self.middle7 = XBlock(728)
        self.middle8 = XBlock(728)

        self.exit1 = SBlock( 728, 728,1024)
        self.exit2 = EBlock(1024,1536,2048)
        self.fc = nn.Linear(2048, num_classes)


    def forward(self,x):

        x = self.entry0(x)    #; print('entry0 ', x.size())
        x = self.entry1(x)    #; print('entry1 ', x.size())
        x = self.entry2(x)    #; print('entry2 ', x.size())
        x = self.entry3(x)    #; print('entry3 ', x.size())
        x = self.middle1(x)   #; print('middle1 ',x.size())
        x = self.middle2(x)   #; print('middle2 ',x.size())
        x = self.middle3(x)   #; print('middle3 ',x.size())
        x = self.middle4(x)   #; print('middle4 ',x.size())
        x = self.middle5(x)   #; print('middle5 ',x.size())
        x = self.middle6(x)   #; print('middle6 ',x.size())
        x = self.middle7(x)   #; print('middle7 ',x.size())
        x = self.middle8(x)   #; print('middle8 ',x.size())
        x = self.exit1(x)     #; print('exit1 ',x.size())
        x = self.exit2(x)     #; print('exit2 ',x.size())

        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, training=self.training, p=0.2)     #
        x = self.fc (x)
        return x #logits


########################################################################################################


def run_check_net():

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size  = 1
    num_classes = 5000
    C,H,W = 3,180,180

    inputs = torch.randn(batch_size,C,H,W)
    labels = torch.randn(batch_size,num_classes)
    in_shape = inputs.size()[1:]


    net = Xception(in_shape=in_shape, num_classes=num_classes)
    net.load_pretrain_pytorch_file(
            '/root/share/data/models/reference/imagenet/xception/caffe-model/inception/xception/xception.keras.convert.pth',
            skip=['fc.weight'	,'fc.bias']
        )
    net.cuda().train()

    x = Variable(inputs).cuda()
    y = Variable(labels).cuda()
    logits = net.forward(x)
    probs  = F.softmax(logits)

    loss = F.binary_cross_entropy_with_logits(logits, y)
    loss.backward()

    print(type(net))
    print(net)

    print('probs')
    print(probs)

    #merging
    # net.eval()
    # net.merge_bn()


########################################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_net()

