"""
Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    - Implemented separate learning rates and optimizers for last (fc) layer vs. all other layers
    - Added/modified command line arguments
        - --algo indicates the asymmetric feedback algorithm to use for non-last layers
        - --last-layer-algo indicates the asymmetric feedback algorithm to use for the last layer
        - --batch-manhattan activates the batch manhattan SGD optimizer for all layers except last fc layer
        - --last-layer-batch-manhattan activates the batch manhattan SGD optimizer for the last layer
        - --learning-rate no longer applies to last layer; --last-layer-learning-rate controls it
        - --lr-decay sets num of epochs by which to decrease both lr's 10-fold
Reference for training settings:
    - https://github.com/pytorch/examples/tree/master/imagenet
"""

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.optim
import bm_sgd


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--algo', default='sign_symmetry', type=str, metavar='ALGO',
                    help='algorithm for asymmetric feedback weight; ' +
                         'options: sign_symmetry, feedback_alignment, sham, or None; ' +
                         'None indicates to use unmodified torchvision reference model ' +
                         '(default: sign_symmetry)')
parser.add_argument('--last-layer-algo', '--lalgo', default='None', type=str, metavar='ALGO',
                    help='algorithm for asymmetric feedback weight for last layer; ' +
                         'options: sign_symmetry, feedback_alignment, sham, or None; ' +
                         'None indicates to use unmodified torch.nn module ' +
                         'disabled if --algo is None (default: None)')
parser.add_argument('--batch-manhattan', '--bm', dest='batch_manhattan',
                    action='store_true', help='use batch manhattan')
parser.add_argument('--last-layer-batch-manhattan', '--lbm', dest='last_layer_batch_manhattan',
                    action='store_true', help='use batch manhattan')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--llr', '--last-layer-learning-rate', default=0.1, type=float,
                    metavar='LLR', help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=10, type=int, metavar='LRD',
                    help='number of epochs after which lr is decreased 10x (default: 10)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0


def main():
    global args, best_prec1, lr_decay
    args = parser.parse_args()
    lr_decay = args.lr_decay

    if args.algo == 'None':
        import torchvision.models as models
    else:
        import models

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.algo == 'None':
        if args.pretrained:
            print("=> using pre-trained reference model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating reference model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()
    else:
        assert args.arch.startswith('resnet') or args.arch.startswith('alexnet'),\
            'only resnets or alexnet supported'
        if args.pretrained:
            raise ValueError("Using non-standard models but pretrained set to True")
        print("=> creating asymmetric feedback model '{}'".format(args.arch) +
              "with non-last layer af_algo '{}' and last layer af_algo '{}'".
              format(args.algo, args.last_layer_algo))
        model = models.__dict__[args.arch](af_algo=args.algo,
                                           last_layer_af_algo=args.last_layer_algo)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if args.gpu is not None:
        if args.arch.startswith('resnet'):
            model_last_parameters = list(model.fc.parameters())
        elif args.arch.startswith('alexnet'):
            model_last_parameters = list(model.classifier[-1].parameters())
    else:
        if args.arch.startswith('resnet'):
            model_last_parameters = list(model.module.fc.parameters())
        elif args.arch.startswith('alexnet'):
            if args.distributed:
                model_last_parameters = list(model.module.classifier[-1].parameters())
            else:
                model_last_parameters = list(model.classifier[-1].parameters())

    model_nonlast_parameters = [p for p in model.parameters() if
                                not any([p is p_last for p_last in model_last_parameters])]

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    if args.batch_manhattan:
        print('using batch manhattan SGD for non-last layers, lr = %.0e' % args.lr)
        optimizer1 = bm_sgd.BMSGD(model_nonlast_parameters, args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print('not using batch manhattan SGD for non-last layers, lr = %.0e' % args.lr)
        optimizer1 = torch.optim.SGD(model_nonlast_parameters, args.lr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)
    if args.last_layer_batch_manhattan:
        print('using batch manhattan SGD for last layer, lr = %.0e' % args.llr)
        optimizer2 = bm_sgd.BMSGD(model_last_parameters, args.llr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print('not using batch manhattan SGD for last layer, lr = %.0e' % args.llr)
        optimizer2 = torch.optim.SGD(model_last_parameters, args.llr,
                                     momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer1.load_state_dict(checkpoint['optimizer1'])
            optimizer2.load_state_dict(checkpoint['optimizer2'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer1, epoch, args.lr)
        adjust_learning_rate(optimizer2, epoch, args.llr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer1, optimizer2, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer1, optimizer2, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr0):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr0 * (0.1 ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
