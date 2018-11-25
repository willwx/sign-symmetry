"""
Slight modification of torch.optim.SGD
    - `d_p = p.grad.data` is replaced by `d_p = p.grad.data.sign()` in `step()` to implement batch manhattan
"""

import torch
import torch.optim


class NSCSGD(torch.optim.SGD):
    def __init__(self, *args, lbound=1e-10, **kwargs):
        super(NSCSGD, self).__init__(*args, **kwargs)
        self._lbound = abs(float(lbound))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
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

                # added rountine for preventing sign change
                pmask = p.data >= 0
                p_ori = torch.tensor(p.data)
                p.data.add_(-group['lr'], d_p)
                flipmask = (p_ori.sign() * p.data.sign()) <= 0
                p.data.masked_fill_(pmask & flipmask, self._lbound)
                p.data.masked_fill_(~pmask & flipmask, -self._lbound)

        return loss
