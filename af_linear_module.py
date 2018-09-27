"""
Modified from https://github.com/L0SG/feedback-alignment-pytorch/blob/master/lib/fa_linear.py
"""

import math
import torch.nn as nn
from af_linear_function import AsymmetricFeedbackLinearFunc


class AsymmetricFeedbackLinear(nn.Linear):

    def __init__(self,  *args, algo='sign_symmetry', **kwargs):
        """
        :param algo: options: ('sign_symmetry', 'feedback_alignment', 'sham')
        """
        assert algo in ('sign_symmetry', 'feedback_alignment', 'sham'), 'algorithm %s is not supported' % algo
        if 'dilation' in kwargs.keys() and kwargs['dilation'] != 1:
            raise ValueError('dilation is not supported in this implementation of', self.__class__.__name__)
        super(AsymmetricFeedbackLinear, self).__init__(*args, **kwargs)

        # this scale is used to initialize weights in torchvision/nn/modules/linear.py
        self.scale = 1. / math.sqrt(self.weight.size(1))

        self.algo = algo
        feedback_weight = None
        if algo == 'feedback_alignment':
            feedback_weight = self.weight.new_empty(self.weight.shape).detach_()
            # this init formula is used in Linear.reset_parameters() in torchvision/nn/modules/linear.py
            feedback_weight.data.uniform_(-self.scale, self.scale)
        self.register_buffer('feedback_weight', feedback_weight)

    def forward(self, input):
        # symmetrical weight for backprop is initialized here
        if self.algo == 'feedback_alignment':
            feedback_weight = self.feedback_weight
        elif self.algo == 'sign_symmetry':
            feedback_weight = self.weight.sign().detach_() * self.scale
        elif self.algo == 'sham':
            feedback_weight = self.weight.detach()
        else:
            raise RuntimeError('unsupported algorithm for %s' % self.__class__.__name__)

        return AsymmetricFeedbackLinearFunc.apply(input, self.weight, feedback_weight, self.bias)



