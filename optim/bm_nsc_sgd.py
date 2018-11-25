"""
Slight modification of torch.optim.SGD
    - `d_p = p.grad.data` is replaced by `d_p = p.grad.data.sign()` in `step()` to implement batch manhattan
"""

import torch
from torch.optim.optimizer import Optimizer, required


class BMNSC_SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 batch_manhattan=False, no_sign_change=False, lower_bound=1e-10):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        batch_manhattan = bool(batch_manhattan)
        no_sign_change = bool(no_sign_change)
        lower_bound = abs(float(lower_bound))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        batch_manhattan=batch_manhattan, no_sign_change=no_sign_change,
                        lower_bound=lower_bound)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            bm = group['batch_manhattan']
            nsc = group['no_sign_change']
            lower_bound = group['lower_bound']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if bm:
                    d_p = d_p.sign_()    # single line change
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if nsc:
                    # added rountine for preventing sign change
                    pmask = p.data >= 0
                    p_ori = torch.tensor(p.data)
                p.data.add_(-group['lr'], d_p)
                if nsc:
                    flipmask = (p_ori.sign() * p.data.sign()) <= 0
                    p.data.masked_fill_(pmask & flipmask, lower_bound)
                    p.data.masked_fill_(~pmask & flipmask, -lower_bound)

        return loss
