"""
Supports asymmetric feedback algorithms using the autograd Function defined in af_conv2d_function
    - sign_symmetry: feedback weights share sign of feedforward weights but not magnitude
    - feedback_alignment: random, fixed feedback weights
with a control option
    - sham: uses feedforward weights for feedback as in backprop; should behave just like nn.Conv2d
    - other related algorithms: 'sign_symmetry_random_magnitude', 'feedback_alignment_signed_init'
References:
    - https://github.com/L0SG/feedback-alignment-pytorch/blob/master/lib/fa_linear.py
"""

import torch
import torch.nn as nn
from functional.af_conv2d_function import AsymmetricFeedbackConv2dFunc
import math


class AsymmetricFeedbackConv2d(nn.Conv2d):
    def __init__(self, *args, algo='sign_symmetry', **kwargs):
        assert algo in ('sign_symmetry', 'sign_symmetry_random_magnitude',
                        'feedback_alignment', 'feedback_alignment_signed_init', 'sham'),\
            'algorithm %s is not supported' % algo
        if 'dilation' in kwargs.keys() and kwargs['dilation'] != 1:
            raise ValueError('dilation is not supported in this implementation of', self.__class__.__name__)
        super(AsymmetricFeedbackConv2d, self).__init__(*args, **kwargs)

        # this scale is used to initialize resnet models in torchvision/models/resnet.py
        # here it is used as a heuristic to avoid the exploding gradient problem
        self.scale = math.sqrt(2 / (self.kernel_size[0] * self.kernel_size[1] * self.out_channels))

        self.algo = algo
        # save tensor version of stride & padding for use in ws_conv2d_function
        stride_tensor = torch.Tensor(self.stride).type(torch.int)
        padding_tensor = torch.Tensor(self.padding).type(torch.int)
        self.register_buffer('stride_tensor', stride_tensor)
        self.register_buffer('padding_tensor', padding_tensor)
        self.reset_feedback_weight()

    def reset_feedback_weight(self):
        feedback_weight = None
        if self.algo in ('feedback_alignment', 'sign_symmetry_random_magnitude'):
            feedback_weight = self.weight.new_empty(self.weight.shape).detach_()
            feedback_weight.data.normal_(0, self.scale)
            if self.algo == 'sign_symmetry_random_magnitude':
                feedback_weight = feedback_weight.abs_()
        if self.algo == 'feedback_alignment_signed_init':
            feedback_weight = self.weight.sign().detach_() * self.scale
        self.register_buffer('feedback_weight', feedback_weight)

    def forward(self, input):
        if self.algo in ('feedback_alignment', 'feedback_alignment_signed_init'):
            feedback_weight = self.feedback_weight
        # symmetrical weight for backprop is initialized here
        elif self.algo == 'sign_symmetry':
            feedback_weight = self.weight.sign().detach_() * self.scale
        elif self.algo == 'sign_symmetry_random_magnitude':
            feedback_weight = self.feedback_weight * self.weight.sign().detach_()
        elif self.algo == 'sham':
            feedback_weight = self.weight.detach()
        else:
            raise RuntimeError('unsupported algorithm for %s' % self.__class__.__name__)

        return AsymmetricFeedbackConv2dFunc.apply(
            input, self.weight, feedback_weight, self.bias, self.stride_tensor, self.padding_tensor)

