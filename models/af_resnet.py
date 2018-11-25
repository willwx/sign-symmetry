"""
Modified from orchvision/models/resnet.py
    - references to nn.Conv2d are replaced by AsymmetricFeedbackConv2d
    - references to nn.Linear are (optionally) replaced by AsymmetricFeedbackLinear
    - calling covention is appended to pass argument 'algo' to AsymmetricFeedbackConv2d and AsymmetricFeedbackLinear
    - reseting of conv layer weight is appended to also reset feedback weight, so that feedback_alignment_signed_init
        will work correctly
    - disabled loading pretrained model
"""

import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
from modules.af_conv2d_module import AsymmetricFeedbackConv2d as AFConv2d
from modules.af_linear_module import AsymmetricFeedbackLinear as AFLinear


__all__ = ['AsymmetricFeedbackResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b12a1ed2d.pth',
# }


def conv3x3(in_planes, out_planes, af_algo, stride=1):
    """3x3 convolution with padding"""
    return AFConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False, algo=af_algo)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, af_algo, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, af_algo=af_algo, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, af_algo=af_algo)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, af_algo, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = AFConv2d(inplanes, planes, kernel_size=1, bias=False, algo=af_algo)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = AFConv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False, algo=af_algo)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = AFConv2d(planes, planes * 4, kernel_size=1, bias=False, algo=af_algo)
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


class AsymmetricFeedbackResNet(nn.Module):

    def __init__(self, block, layers, af_algo, num_classes=1000, last_layer_af_algo=None):
        self.inplanes = 64
        super(AsymmetricFeedbackResNet, self).__init__()
        self.conv1 = AFConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False, algo=af_algo)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], af_algo=af_algo)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, af_algo=af_algo)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, af_algo=af_algo)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, af_algo=af_algo)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if last_layer_af_algo is None or last_layer_af_algo == 'None':
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = AFLinear(512 * block.expansion, num_classes, algo=last_layer_af_algo)

        for m in self.modules():
            if isinstance(m, AFConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.reset_feedback_weight()    # important for algo "feedback_alignment_signed_init" to work correctly
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, af_algo, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                AFConv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False, algo=af_algo),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, af_algo=af_algo, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, af_algo=af_algo))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a AsymmetricFeedbackResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AsymmetricFeedbackResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a AsymmetricFeedbackResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AsymmetricFeedbackResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a AsymmetricFeedbackResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AsymmetricFeedbackResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a AsymmetricFeedbackResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AsymmetricFeedbackResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a AsymmetricFeedbackResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AsymmetricFeedbackResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
