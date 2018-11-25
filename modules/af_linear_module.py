"""
Modified from https://github.com/L0SG/feedback-alignment-pytorch/blob/master/lib/fa_linear.py
"""

import math
import torch.nn as nn
from functional.af_linear_function import AsymmetricFeedbackLinearFunc


class AsymmetricFeedbackLinear(nn.Linear):

    def __init__(self,  *args, algo='sign_symmetry', **kwargs):
        assert algo in ('sign_symmetry', 'sign_symmetry_random_magnitude',
                        'feedback_alignment', 'feedback_alignment_signed_init', 'sham'),\
            'algorithm %s is not supported' % algo
        super(AsymmetricFeedbackLinear, self).__init__(*args, **kwargs)

        # this scale is used to initialize weights in torchvision/nn/modules/linear.py
        self.scale = 1. / math.sqrt(self.weight.size(1))

        self.algo = algo
        feedback_weight = None
        if algo in ('feedback_alignment', 'sign_symmetry_random_magnitude'):
            feedback_weight = self.weight.new_empty(self.weight.shape).detach_()
            if algo == 'sign_symmetry_random_magnitude':
                feedback_weight.data.uniform_(0, self.scale)
            else:
                # this init formula is used in Linear.reset_parameters() in torchvision/nn/modules/linear.py
                feedback_weight.data.uniform_(-self.scale, self.scale)  # * math.sqrt(3) for equal stdev to other algos
        if algo == 'feedback_alignment_signed_init':
            feedback_weight = self.weight.sign().detach_() * self.scale
        self.register_buffer('feedback_weight', feedback_weight)

    def forward(self, input):
        if self.algo == 'feedback_alignment' or self.algo == 'feedback_alignment_signed_init':
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

        return AsymmetricFeedbackLinearFunc.apply(input, self.weight, feedback_weight, self.bias)



