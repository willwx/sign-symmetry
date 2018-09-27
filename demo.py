"""
Helpful script to compare behavior of a sign-symmetry model with that of a standard model
Usage: python demo.py [--algo sign_symmetry] [--bm] [--lbm]
"""

import argparse
import torch


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--sign-symmetry', '-ss', metavar='SS', default=1, type=int,
                    help='use sign symmetry; 0: no, other: yes (default: 1)')
parser.add_argument('--batch-manhattan', '-bm', metavar='BM', default=1, type=int,
                    help='use batch manhattan; 0: no, other: yes (default: 1)')
args = parser.parse_args()

if args.sign_symmetry == 1:
    from models import resnet18
else:
    from torchvision.models import resnet18

if args.batch_manhattan == 1:
    from bm_sgd import BMSGD as SGD
else:
    from torch.optim import SGD


torch.manual_seed(0)  # set random state for deterministic behavior
m = resnet18()
print(m.fc.weight)    # both the torchvision model and the custom model should have the same initial weights

optimizer = SGD(m.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)


# do a single forward/backward pass & update
input = torch.rand(1, 3, 224, 224)
output = m.forward(input)

target = torch.LongTensor(1)
target[0] = 0
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(output, target)

optimizer.zero_grad()
loss.backward()
optimizer.step()


# after a single update, last layer should be exactly the same regardless of whether sign symmetry is used,
#  because sign symmetry does not affect updates in the last layer
print(m.fc.weight)


# after a single update, if batch manhattan is used, many first layer weights should be equal between
#  a sign symmetry model and a standard model, because gradients that have equal signs (but possibly different values)
#  will lead to the same updates. furthermore, these updates will be exactly +/- 0.1 because batch manhattan sets
#  gradients to +/- 1, and learning_rate == 0.1
# if batch manhattan is not used, first layer weights are expected to differ between a sign symmetry model and
#  a standard model
print(m.conv1.weight)
