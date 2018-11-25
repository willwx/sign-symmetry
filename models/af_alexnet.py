"""
Modified from orchvision/models/alexnet.py
    - references to nn.Conv2d are replaced by AsymmetricFeedbackConv2d
    - references to nn.Linear are (partially optionally) replaced by AsymmetricFeedbackLinear
        - changed the first two linear layers to use asymmetric feedback
        - added option for last layer to also use asymmetric feedback
    - calling covention is appended to pass argument 'algo' to AsymmetricFeedbackConv2d and AsymmetricFeedbackLinear
    - architecture changes
        - added batchnorm before every ReLU
        - removed bias before every batchnorm
        - removed Dropout
    - disabled loading pretrained model
"""

import torch.nn as nn
# import torch.utils.model_zoo as model_zoo
from modules.af_conv2d_module import AsymmetricFeedbackConv2d as AFConv2d
from modules.af_linear_module import AsymmetricFeedbackLinear as AFLinear


__all__ = ['AlexNet', 'alexnet']


# model_urls = {
#     'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


class AlexNet(nn.Module):
    """
    Modified AlexNet: Added Batchnorm before each non-linearity, removed bias of the preceding layer, and
    removed dropout
    """

    def __init__(self, af_algo, num_classes=1000, last_layer_af_algo=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            AFConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False, algo=af_algo),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            AFConv2d(64, 192, kernel_size=5, padding=2, bias=False, algo=af_algo),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            AFConv2d(192, 384, kernel_size=3, padding=1, bias=False, algo=af_algo),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            AFConv2d(384, 256, kernel_size=3, padding=1, bias=False, algo=af_algo),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            AFConv2d(256, 256, kernel_size=3, padding=1, bias=False, algo=af_algo),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if last_layer_af_algo is None or last_layer_af_algo == 'None':
            last_layer = nn.Linear(4096, num_classes)
        else:
            last_layer = AFLinear(4096, num_classes, algo=last_layer_af_algo)
        self.classifier = nn.Sequential(
            # nn.Dropout(),    # not that necessary, at least according to Ioffe & Szegedy 2015 <arXiv:1502.03167v3>
            AFLinear(256 * 6 * 6, 4096, bias=False, algo=af_algo),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            AFLinear(4096, 4096, bias=False, algo=af_algo),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            last_layer,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        raise NotImplemented
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
