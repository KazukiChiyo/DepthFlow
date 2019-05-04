# Author: Kexuan Zou
# Date: Mar 22, 2019
# License: MIT

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
# from models.DepthFlowNetS import depthflownets
from backend import Train, Test, MultiScaleEPE, EPE, AdaBound
from datasets import KITTI_noc, utils
from tensorboardX import SummaryWriter
# from models.test import flownets_bn
from models import FlowNetS

data_dir = './KITTI/training'

def checkpoint(state):
    model_out_path = "./ckpts/{}:{}_epoch_{}_epe{:.4f}_df{:.4f}.pth".format(state['name'], state['solver'], state['epoch'], state['epe'], state['div_flow'])
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

parser = argparse.ArgumentParser(description='PyTorch Vogel Training')
parser.add_argument('--workers', default=8, type=int,
    help='number of data loading workers')
parser.add_argument('--n_epochs', default=300, type=int,
    help='number of total epochs')
parser.add_argument('--batch_size', default=8, type=int,
    help='batch size')
parser.add_argument('--n_scales', default=5, type=int,
    help='number of different scales for MultiScaleEPE')
parser.add_argument('--l_weights', default=0.005, type=float,
    help='initial weight loss for MultiScaleEpe')
parser.add_argument('--div_flow', default=20, type=float,
    help='flow normalizing factor')
parser.add_argument('--solver', default='adam', choices=['adam','adabound', 'sgd'],
    help='solver algorithm, one of adam, adabound or sgd')
parser.add_argument('--lr', default=1e-4, type=float,
    help='(initial) learning rate')
parser.add_argument('--alpha', default=0.9, type=float,
    help='alpha term for Adam or AdaBound')
parser.add_argument('--beta', default=0.999, type=float,
    help='beta term for Adam or AdaBound')
parser.add_argument('--final_lr', default=0.1, type=float,
    help='final (SGD) learning rate if AdaBound is used')
parser.add_argument('--weight_decay', default=4e-4, type=float,
    help='weight decay')
parser.add_argument('--bias_decay', default=0, type=float,
    help='bias decay')
parser.add_argument('--milestones', default=[100, 150, 200],
    nargs='*', type=int, help='epochs at which learning rate decays')
parser.add_argument('--lr_decay', default=0.5, type=float,
    help='factor of learning rate decay')
parser.add_argument('--depth', default=True, type=bool,
    help='true if depth information is needed')
args = parser.parse_args()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('No GPU found, running on CPU')
        device = torch.device("cpu")
        pin_memory = False
    else:
        device = torch.cuda.current_device()
        print('Using ' + torch.cuda.get_device_name(device))
        pin_memory = True

    print('--- Loading datasets ---')
     # Data loading code
    #TODO
    rgb_transform = transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    # depth transform will be ignored if arg.depth=False
    depth_transform=transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[255]), #???
        transforms.Normalize(mean=[0.5], std=[1])
    ])


    target_transform = transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])

    # not all utils transform support depth!
    co_transform = utils.Compose([
        utils.RandomCrop((320,448)),
        utils.RandomVerticalFlip(),
        utils.RandomHorizontalFlip()
    ])


    train_loader, valid_loader = KITTI_noc(dir=data_dir, batch_size=args.batch_size,
        rgb_transform=rgb_transform, target_transform=target_transform, co_transform=co_transform,
                                           depth_transform=depth_transform,
        split=0.9, num_workers=args.workers, pin_memory=pin_memory,depth=args.depth)

    print('--- Building model ---')
    cudnn.benchmark = True
    if args.depth:
        in_channels = 8
    else:
        in_channels = 6
    model = FlowNetS(in_channels=in_channels, batch_norm=True).to(device)
    model_name = 'FlowNetS_depth'
    criterion = MultiScaleEPE(n_scales=args.n_scales, l_weight=args.l_weights)
    metric = EPE(div_flow=args.div_flow)

    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
        {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    if args.solver == 'adam':
        optimizer = optim.Adam(param_groups, lr=args.lr, betas=(args.alpha, args.beta))
    elif args.solver == 'adabound':
        optimizer = AdaBound(param_groups, lr=args.lr, betas=(args.alpha, args.beta), final_lr=args.final_lr)
    elif args.solver == 'sgd':
        optimizer = optim.SGD(param_groups, lr=args.lr, momentum=args.alpha)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay)

    train = Train(model=model, data_loader=train_loader, optim=optimizer, criterion=criterion, metric=metric, device=device)
    train_writer = SummaryWriter(log_dir='./logs/train')
    valid = Test(model=model, data_loader=valid_loader, metric=metric, device=device)
    valid_writer = SummaryWriter(log_dir='./logs/valid')
    best_epe = 2**32

    for epoch in range(1, args.n_epochs + 1):
        scheduler.step()
        train_loss, train_epe = train.run_epoch()
        print(">>>> Epoch {}: loss: {:.4f}, epe: {:.4f}".format(epoch, train_loss, train_epe))
        train_writer.add_scalar('avg loss', train_loss, epoch)
        train_writer.add_scalar('avg epe', train_epe, epoch)
        if epoch%5 == 0:
            valid_epe = valid.run_epoch()
            print(">>>> Validation: epe: {:.4f}".format(valid_epe))
            valid_writer.add_scalar('avg epe', valid_epe, epoch)
            if valid_epe < best_epe:
                checkpoint(state={
                    'name': model_name,
                    'solver': args.solver,
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'epe': valid_epe,
                    'div_flow': args.div_flow,
                })
                best_epe = valid_epe
