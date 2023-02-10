import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import re
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

models_dir = 'imagenet_pretrained'
model_name = {
    'resnet18': 'resnet18_abri.pth',
    'resnet34': 'resnet34_abri.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fusion = 'batch', fus_weight = 0.5):
        super(BasicBlock, self).__init__()
        self.fusion = fusion
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn1_addition = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if fusion == 'batch':
            self.fusion_weight1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        else:
            self.fusion_weight1 = nn.Parameter(torch.FloatTensor(1,planes,1,1), requires_grad=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_addition = nn.BatchNorm2d(planes)

        if fusion == 'batch':
            self.fusion_weight2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        else:
            self.fusion_weight2 = nn.Parameter(torch.FloatTensor(1,planes,1,1), requires_grad=True)
        
        self.fusion_weight1.data.fill_(fus_weight)
        self.fusion_weight2.data.fill_(fus_weight)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        
        out1 = self.bn1(out)
        out2 = self.bn1_addition(out)
        if self.fusion == 'batch':
            out = self.fusion_weight1 * out1 + (1 - self.fusion_weight1) * out2
        else:
            out = torch.mul(self.fusion_weight1, out1) + torch.mul(1 - self.fusion_weight1, out2)

        out = self.relu(out)

        out = self.conv2(out)

        out1 = self.bn2(out)
        out2 = self.bn2_addition(out)
        if self.fusion == 'batch':
            out = self.fusion_weight2 * out1 + (1 - self.fusion_weight2) * out2
        else:
            out = torch.mul(self.fusion_weight2, out1) + torch.mul(1 - self.fusion_weight2, out2)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class DownSample(nn.Module):
    def __init__(self, inplanes, outplanes, stride, fusion = 'batch', fus_weight = 0.5):
        super(DownSample, self).__init__()
        self.fusion = fusion
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(outplanes)
        self.bn_addition = nn.BatchNorm2d(outplanes)
        if fusion == 'batch':
            self.fusion_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        else:
            self.fusion_weight = nn.Parameter(torch.FloatTensor(1,outplanes,1,1), requires_grad=True)
        self.fusion_weight.data.fill_(fus_weight)
    
    def forward(self, x):
        out = self.conv(x)
        out1 = self.bn(out)
        out2 = self.bn_addition(out)
        if self.fusion == 'batch':
            out = self.fusion_weight * out1 + (1 - self.fusion_weight) * out2
        else:
            out = torch.mul(self.fusion_weight, out1) + torch.mul(1 - self.fusion_weight, out2)
        
        return out




class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channel = 3, fusion = 'batch', fus_weight=0.5):
        super(ResNet, self).__init__()
        self.fusion = fusion
        self.fus_weight = fus_weight
        self.inplanes = 64
        self.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_addition = nn.BatchNorm2d(64)
        if fusion == 'batch':
            self.fusion_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
        else:
            self.fusion_weight = nn.Parameter(torch.FloatTensor(1,64,1,1), requires_grad=True)
        self.fusion_weight.data.fill_(fus_weight)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownSample(self.inplanes, planes * block.expansion, stride=stride, fusion = self.fusion, fus_weight = self.fus_weight)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.fusion, self.fus_weight))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fusion = self.fusion, fus_weight = self.fus_weight))

        return nn.Sequential(*layers)

    def forward(self, x):
        

        x = self.conv1(x)

        x1 = self.bn1(x)
        x2 = self.bn1_addition(x)
        if self.fusion == 'batch':
            x = self.fusion_weight * x1 + (1 - self.fusion_weight) * x2
        else:
            x = torch.mul(self.fusion_weight, x1) + torch.mul(1 - self.fusion_weight, x2)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # print(x.shape)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x
    
    def forward_layer0(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        return x

    def forward_layer1(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        return x
    
    def forward_layer2(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_layer3(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def naive_bn_random(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.running_var.data.fill_(1)
                m.running_mean.data.zero_()
    
    def better_naive_bn_random(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                n = m.weight.shape[0]
                for i in range(n):
                    if(torch.abs(m.weight.data[i])< 1e-6):
                        m.weight.data[i] = 1
                        m.bias.data[i] = 0


def resnet18(imagenet_pretrained=True, fusion = 'batch', fus_weight = 0.5,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        imagenet_pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], fusion = fusion, fus_weight = fus_weight, **kwargs)
    if imagenet_pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])), strict=False)
    return model

def resnet34(imagenet_pretrained=True, fusion = 'batch', fus_weight = 0.5,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        imagenet_pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], fusion = fusion, fus_weight = fus_weight, **kwargs)
    if imagenet_pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])), strict=False)
    return model






    