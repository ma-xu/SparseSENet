"""SE-ResNet in PyTorch
Based on preact_resnet.py

Author: Xu Ma.
Date: Apr/15/2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'SEResNet152']

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16,activation='Sigmoid',factor=1):
        super(SELayer, self).__init__()
        self.factor = factor
        self.activation = activation
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        if str.lower(activation) == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        if str.lower(self.activation) == 'sigmoid':
            y = self.act(self.factor*y).view(b, c, 1, 1)
        else:
            y = (self.act(self.factor * y)*c).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16,activation='Sigmoid',factor=1):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = SELayer(planes,reduction,activation,factor)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16,activation='Sigmoid',factor=1):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = SELayer(self.expansion*planes, reduction,activation,factor)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class SEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=1000,reduction=16,activation='Sigmoid',factor=1):
        super(SEResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction,
                                       activation=activation,factor =factor)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction,
                                       activation=activation,factor =factor)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction,
                                       activation=activation,factor =factor)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction,
                                       activation=activation,factor =factor)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    #block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks,stride,reduction,activation,factor):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction,activation,factor))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SEResNet18(num_classes=1000,activation='Sigmoid',factor=1):
    return SEResNet(SEPreActBlock, [2,2,2,2],num_classes,activation=activation,factor=factor)


def SEResNet34(num_classes=1000,activation='Sigmoid',factor=1):
    return SEResNet(SEPreActBlock, [3,4,6,3],num_classes,activation=activation,factor=factor)


def SEResNet50(num_classes=1000,activation='Sigmoid',factor=1):
    return SEResNet(SEPreActBootleneck, [3,4,6,3],num_classes,activation=activation,factor=factor)


def SEResNet101(num_classes=1000,activation='Sigmoid',factor=1):
    return SEResNet(SEPreActBootleneck, [3,4,23,3],num_classes,activation=activation,factor=factor)


def SEResNet152(num_classes=1000,activation='Sigmoid',factor=1):
    return SEResNet(SEPreActBootleneck, [3,8,36,3],num_classes,activation=activation,factor=factor)


def mydemo():
    net = SEResNet50(activation='Softmax',factor=2)
    y = net((torch.randn(10,3,32,32)))
    print(y.size())


# mydemo()