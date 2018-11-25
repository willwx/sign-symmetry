This pacakge implements PyTorch ResNet & AlexNet models that support
 two asymmetric feedback learning algorithms:
 - sign-symmetry
 ([Liao, Leibo, & Poggio, 2016](https://dl.acm.org/citation.cfm?id=3016156));
 - feedback alignment ([Lillicrap et al., 2016](https://www.nature.com/articles/ncomms13276)).

To use different weights in forward and backward passes, we needed to modify
 the `backward()` routine.
Since the interface to `backward()` in `torch.nn.Conv2d` is not exposed
 to Python, it is necessary to implement a custom convolution `autograd.Function`.
This is achieved by borrowing routines from `torch.legacy.nn.SpatialConvolution`
 that expose separate forward/backward interfaces but still use C libraries
 to carry out the computations efficiently.
Then, a custom Conv2d module is defined with the custom Conv2d Function and
 used to construct ResNet models by customizing `torchvision/models/resnet.py`
 to replace standard Conv2d layers with the custom version. 
In this implementation, batch-norm and pooling layers continue to use
 backprop, but in our experience this does not affect performance. 

In addition, a training script for ImageNet is included modified from
 [this one](https://github.com/pytorch/examples/blob/master/imagenet/main.py).
The command line arguments `--algo` and `--lalgo` control the algorithm used
 for convolutional layers and the last layer, respectively (only
 implemented for ResNet and AlexNet).
By default, sign-symmetry is used for the convlutional layers and
 backpropagation is used for the last layer. 
Another option is added to use batch-manhattan SGD
 (Liao, Leibo, & Poggio, 2016) in place of standard SGD
 because it was previously found to improve training with sign-symmetry.
It is implemented by extending `torch.optim.SGD`. 
 
It should be relatively straightforward to extend this package to support
 other network architectures.  