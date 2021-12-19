from pprint import pprint
from torchsummary import summary
import pandas as pd
import pickle
#import wandb
import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models
from models.quan_ops import SwitchBatchNorm2d
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, setup_gpus, save_checkpoint
from utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--results-dir', default='./results', help='results dir')
parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='resnet18', help='model architecture')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', help='optimizer function used')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', default='100,150,180', help='lr decay steps')
parser.add_argument('--weight-decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default=None, help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--bit_width_list', default='4', help='bit width list')
args = parser.parse_args()


def main():
    hostname = socket.gethostname()
    setup_logging(os.path.join('/home/tmddnjs33/data/emb/embedded_system_assign', args.results_dir, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)
    device = torch.device("cpu")

    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, args.train_split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    bit_width_list = list(map(int, args.bit_width_list.split(',')))
    bit_width_list.sort()
    model = models.__dict__[args.model](bit_width_list, train_data.num_classes).cpu()
    numpy_model = models.__dict__[args.model+'_np'](bit_width_list, train_data.num_classes)

    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(best_gpu))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            #checkpoint = torch.load(args.pretrain, map_location='cuda:{}'.format(best_gpu))
            checkpoint = torch.load(args.pretrain, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])

            # print parameters
            row_list = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad: continue
                param = parameter.numel()
                row_list.append([name, param])
            df = pd.DataFrame(row_list, columns = ["Modules", "Parameters"])
            pprint(df.to_string())
            summary(model, (3, 32, 32))

            
        else:
            raise ValueError('Pretrained model path error!')
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    print('Numpy Test start!')
    numpy_test(model, numpy_model)


def numpy_test(model, np_model):
    weight_list = {}

    print('Load weight start!')
    for i, parameter in enumerate(model.named_parameters()):
        name, param = parameter
        if param.requires_grad: 
            param_np = param.detach().numpy()
        else:
            param_np = param.numpy()

        weight_list[name] = param_np

    print('Load weight done!')

    with open("testset/data.pkl", "rb") as f:
        print('Load test data set!')
        input, target = pickle.load(f)
        print('Load test data set done!')
        print('Forward calculation start!')
        out = np_model.forward(input, weight_list)
        print(out)
        print('Forward calculation done!')

    exit()

if __name__ == '__main__':
    main()
