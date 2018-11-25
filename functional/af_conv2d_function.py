"""
A conv2d autograd Function that supports different feedforward and feedback weights
Uses the same C functions used in the legacy SpatialConvolution module to implement actual computation
References:
    - https://github.com/L0SG/feedback-alignment-pytorch/blob/master/lib/fa_linear.py
    - https://pytorch.org/docs/master/notes/extending.html
    - torch/legacy/nn/SpatialConvolution.py
"""

import torch.autograd as autograd
from torch._thnn import type2backend


class AsymmetricFeedbackConv2dFunc(autograd.Function):

    @staticmethod
    def forward(context, input, weight, weight_feedback, bias, stride, padding):
        _backend = type2backend[input.type()]
        input = input.contiguous()
        output = input.new()
        finput = input.new()
        fgradInput = input.new()

        _backend.SpatialConvolutionMM_updateOutput(
            _backend.library_state,
            input,
            output,
            weight,
            bias,
            finput,
            fgradInput,
            weight.shape[2], weight.shape[3],
            int(stride[0]), int(stride[1]),
            int(padding[0]), int(padding[1])
        )

        context.save_for_backward(input, weight, weight_feedback, bias, stride, padding, finput, fgradInput)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_feedback, bias, stride, padding, finput, fgradInput = context.saved_variables
        grad_input = grad_weight = grad_bias = None

        _backend = type2backend[grad_output.type()]
        grad_output = grad_output.contiguous()
        ksize = tuple(weight.shape[-2:])

        if context.needs_input_grad[0]:
            grad_input = input.new()

            _backend.SpatialConvolutionMM_updateGradInput(
                _backend.library_state,
                input,
                grad_output,
                grad_input,
                weight_feedback,
                finput,
                fgradInput,
                ksize[0], ksize[1],
                int(stride[0]), int(stride[1]),
                int(padding[0]), int(padding[1])
            )

        if context.needs_input_grad[1] or context.needs_input_grad[3]:
            grad_weight = weight.grad or weight.new_zeros(weight.shape)
            grad_bias = None if bias is None else bias.grad or bias.new_zeros(bias.shape)

            _backend.SpatialConvolutionMM_accGradParameters(
                _backend.library_state,
                input,
                grad_output,
                grad_weight,
                grad_bias,
                finput,
                fgradInput,
                ksize[0], ksize[1],
                int(stride[0]), int(stride[1]),
                int(padding[0]), int(padding[1]),
                1
            )

        return grad_input, grad_weight, None, grad_bias, None, None
