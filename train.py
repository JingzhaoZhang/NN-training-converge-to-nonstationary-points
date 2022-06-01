import argparse
import os
import sys
import random
import shutil
import time
import copy
import warnings
import errno
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models import *
from utils import *
import warnings
warnings.filterwarnings("ignore")

null = open(os.devnull,'wb')

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument( '-ls', '--lr_schedule', default='piecewise', type=str,
                    help='piecewise | cosine | constant')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 1/10 iteration)')
parser.add_argument('-ef', '--eval-freq', default=5, type=int,
                    metavar='N', help='evaluation frequency (default: 1 / 5 epoch)')
parser.add_argument('-sf', '--stat-freq', default=2000, type=int,
                    metavar='N', help='stat evaluation frequency (default: 1 / 2000 iteration)')

parser.add_argument('-suppress', action='store_true')

parser.add_argument('-save_sharpness', action='store_true')
parser.add_argument('-sharpness_batches', type=int, default=10)
parser.add_argument('-save_noise', action='store_true')
parser.add_argument('--save-dir', type=str,  default='default',
                    help='path to save the final model')
parser.add_argument('--pretrain_path', type=str,  default='',
                    help='path to save the final model')
parser.add_argument('-noise_size', type=int, default=10)
parser.add_argument('--epoch_interval', '-ei', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')



best_acc1 = 0

def load_model(path, model, optimizer):
    print("=> loading checkpoint '{}'".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(path))

    
    
def main():
    
    args = parser.parse_args()

    if args.suppress:
        sys.stderr = null
        
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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


        
        
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, log_train_file, log_valid_file, log_sharp_file, log_noise_file

    args.gpu = gpu
    
    
    save_dir = args.save_dir + '/'
    log_train_file = save_dir + 'train.csv'
    log_valid_file = save_dir + 'valid.csv'
    log_sharp_file = save_dir + 'sharpness.csv'
    log_noise_file = save_dir + 'noise.csv'

    try:
        print('creating directory: %s' % save_dir)
        os.makedirs(save_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
            
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
#     if args.pretrained:
#         print("=> using pre-trained model '{}'".format(args.arch))
#         model = models.__dict__[args.arch](pretrained=True)
#     else:
#         print("=> creating model '{}'".format(args.arch))
#         model = models.__dict__[args.arch]()

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError
        
        
      
        
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


            
#     weight_names, weights = param_weights(model)

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf, open(log_sharp_file, 'w') as log_sf, open(log_noise_file, 'w') as log_nf:
        log_tf.write('epoch,loss,accu1\n')
        log_nf.write('epoch,sto_grad_norm,stograd_linf,noisenorm,gradnorm,l1norm,linfnorm, update_size, change_in_grad_sq\n')
#         log_sf.write('epoch,sharpness,' +','.join(weight_names) + '\n')
        log_vf.write('epoch,valloss,valaccu\n')
        log_sf.write('epoch,sharpness, dir_sharpness' + '\n')

            
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    
    if args.lr_schedule == "cosine":
        print('using cosine with total step %d' % args.epochs * len(train_loader))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                    args.epochs * len(train_loader), eta_min=args.lr * 0.01)    
 

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
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
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), prefetch_factor=4,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    stats_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), prefetch_factor=4,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, persistent_workers=True)

    
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
        validate(val_loader, model, criterion, args)
        return
    pretrained = bool(args.pretrain_path)
    for epoch in range(args.start_epoch, args.epochs):
        
        
        if args.pretrain_path and epoch % args.epoch_interval == 0:
            pretrain_path = os.path.join(args.pretrain_path, args.arch+'_%d.pth' % (epoch+1))
            load_model(pretrain_path, model, optimizer)

        if args.distributed:
            train_sampler.set_epoch(epoch)
            
            
        if args.lr_schedule == "piecewise":
            adjust_learning_rate(optimizer, epoch, args)
        elif args.lr_schedule == "cosine":
            scheduler.step()

        # train for one epoch
        train(train_loader, stats_loader, model, criterion, optimizer, epoch, args, pretrained=pretrained)

        
        # evaluate on validation set
        if epoch % args.eval_freq == 0:
            acc1 = validate(epoch, val_loader, model, criterion, args)

        # if (not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0)) and (epoch % args.epoch_interval == 0):
        #     save_stats(stats_loader, copy.deepcopy(model), criterion, optimizer, epoch, args)

            
            
def compute_grad_epoch(dataloader, model, criterion, optimizer, epoch, args):
    model.train()
    model.zero_grad()
    
    for i in range(args.noise_size):
        images, target = next(dataloader)        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        
        output = model(images)
        loss = criterion(output, target) / args.noise_size

        # measure accuracy and record loss

        # compute gradient and do SGD step
        loss.backward()


        if i == args.noise_size - 1:
            break
            
    true_grad = {}
    clone_grad(model, true_grad)
    model.zero_grad()
    return true_grad        


            
def compute_sto_grad_norm(dataloader, model, criterion, optimizer, epoch, args, prev_true_grad):
    noise_sq = []
    stograd_sq = []
    stograd_linf = []

    # Turn on training mode which enables dropout.
    model.train()
    true_grads = compute_grad_epoch(dataloader, model, criterion, optimizer, epoch, args)
<<<<<<< HEAD
    if not prev_true_grad:
        prev_true_grad = true_grads

=======
>>>>>>> bc48dcbaad3b7c0a0c30f03fa6d45e0853f187f4
    grad_change_sq, _, _ = compute_noise(true_grads, prev_true_grad)
    gradnorm_sq = compute_norm(true_grads) 
    true_gradnorml1 = compute_l1norm(true_grads) 
    true_gradnormlinf = compute_linfnorm(true_grads) 

    for i in range(args.noise_size):
        images, target = next(dataloader)
        # measure data loading time

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        model.zero_grad()
        output = model(images)
        loss = criterion(output, target) 

        # measure accuracy and record loss

        # compute gradient and do SGD step
        loss.backward()

        sto_grads = {}
        clone_grad(model, sto_grads)
        instance_noisesq, instance_gradsq, instance_gradlinf = compute_noise(sto_grads, true_grads)
        noise_sq.append(instance_noisesq)
        stograd_sq.append(instance_gradsq)
        stograd_linf.append(instance_gradlinf)

        if i == args.noise_size - 1:
            break

    model.zero_grad()
    return noise_sq, stograd_sq, stograd_linf, gradnorm_sq, true_gradnorml1, true_gradnormlinf, grad_change_sq    


def save_stats(stats_loader, model, criterion, optimizer, epoch, args, prev_true_grad, update_size):
    # switch to train mode
    model.train()
    stats_iterator = iter(stats_loader)

    if args.save_sharpness:
        print("Saving sharpness")
        model.zero_grad()
        dir_sharpness = dir_hessian(model, stats_iterator, criterion, args.sharpness_batches)
<<<<<<< HEAD
        stats_iterator = iter(stats_loader)

        model.zero_grad()
        sharpness = eigen_hessian(model, stats_iterator, criterion, args.sharpness_batches)
        stats_iterator = iter(stats_loader)
=======
        model.zero_grad()
        sharpness = eigen_hessian(model, stats_iterator, criterion, args.sharpness_batches)
>>>>>>> bc48dcbaad3b7c0a0c30f03fa6d45e0853f187f4

#         weight_names, weights = param_weights(model)
#         weights_str = ['%4.4f' % w for w in weights]
        

    if args.save_noise:
        print("Saving noise level")
        model.zero_grad()
        # true_gradnorm, sto_grad_norm, sto_noise_norm, true_gradnorml1, true_gradnormlinf = 0,0,0, 0, 0
        noise_sq, stograd_sq, stograd_linf, gradnorm_sq, true_gradnorml1, true_gradnormlinf, grad_change_sq = compute_sto_grad_norm(stats_iterator, model, criterion, 
<<<<<<< HEAD
                                                                                                            optimizer, epoch, args, prev_true_grad)
        sto_grad_norm = np.mean(stograd_sq)
        sto_noise_norm = np.mean(noise_sq)
        stograd_linf = np.mean(stograd_linf)
        
=======
                                                                                                                    optimizer, epoch, args, prev_true_grad)
        sto_grad_norm = np.mean(stograd_sq)
        sto_noise_norm = np.mean(noise_sq)
        stograd_linf = np.mean(stograd_linf)
             
>>>>>>> bc48dcbaad3b7c0a0c30f03fa6d45e0853f187f4
    if args.save_sharpness:
        with open(log_sharp_file, 'a') as log_vf:
            log_vf.write('{epoch},{sharpness: 8.5f},{dir_sharpness: 8.5f},'.format(epoch=epoch, sharpness=sharpness, dir_sharpness=dir_sharpness) + '\n')     
  
    if args.save_noise:
        with open(log_noise_file, 'a') as log_tf:
            log_tf.write('{epoch},{sto_grad_norm:3.3f},{stograd_linf:3.3f},{noisenorm:3.3f},{gradnorm:3.3f},{l1norm:3.3f},{linfnorm:3.3f},{update_size:3.3f},{grad_change_sq:3.3f}\n'.format(
                epoch=epoch,
                gradnorm=gradnorm_sq, sto_grad_norm=sto_grad_norm, stograd_linf=stograd_linf,
                noisenorm=sto_noise_norm, l1norm=true_gradnorml1, linfnorm=true_gradnormlinf, 
                update_size=update_size, grad_change_sq=grad_change_sq))
            

def train(train_loader, stats_loader, model, criterion, optimizer, epoch, args, pretrained=False):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
                
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        prev_true_grad, update_size = None, 0

        if not pretrained:
            if args.save_noise and i % args.stat_freq == 0:

                print("saving stat info before backward")
                prev_true_grad = compute_grad_epoch(iter(stats_loader), model, criterion, optimizer, epoch, args)


            loss.backward()

            if args.save_noise and i % args.stat_freq == 0:

                print("saving stat info before update")
                update_direction = {}
                clone_grad(model, update_direction)
                update_size = compute_norm(update_direction)**0.5 * optimizer.param_groups[0]['lr']

            optimizer.step()
        



        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.save_noise and (pretrained or i % args.stat_freq == 0):
            save_stats(stats_loader, copy.deepcopy(model), criterion, optimizer, epoch, args, prev_true_grad, update_size)

        if pretrained:
            break


    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        with open(log_train_file, 'a') as log_vf:
            log_vf.write('{epoch},{loss: 8.5f},{accu: 8.5f}\n'.format(epoch=epoch, loss=losses.avg, accu=top1.avg))     

            

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)


        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
        with open(log_valid_file, 'a') as log_vf:
            log_vf.write('{epoch},{loss: 8.5f},{accu: 8.5f}\n'.format(epoch=epoch, loss=losses.avg, accu=top1.avg))     

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // int(args.epochs / 3)))
    print("Epoch %d adjusted lr to be %f" % (epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':

    main()