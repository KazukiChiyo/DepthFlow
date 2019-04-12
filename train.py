# Author: Kexuan Zou
# Date: Mar 22, 2019
# License: MIT

import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from models import FlowNetS
from backend import Train, Test, MultiScaleEPE, EPE
from datasets import KITTI_noc, utils
from tensorboardX import SummaryWriter

data_dir = './KITTI/training'

def checkpoint(state):
    model_out_path = "./ckpts/{}_epoch_{}_loss_{:.4f}_epe{:.4f}.pth".format(state['name'], state['epoch'], state['loss'], state['epe'])
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

parser = argparse.ArgumentParser(description='PyTorch Vogel Training')
parser.add_argument('--workers', default=8, type=int,
    help='number of data loading workers')
parser.add_argument('--n_epoches', default=300, type=int,
    help='number of total epoches')
parser.add_argument('--batch_size', default=8, type=int,
    help='batch size')
parser.add_argument('--lr', default=1e-4, type=float,
    help='learning rate')
parser.add_argument('--n_scales', default=5, type=int,
    help='number of different scales for MultiScaleEPE')
parser.add_argument('--l_weights', default=0.32, type=float,
    help='initial weight loss for MultiScaleEpe')
parser.add_argument('--div_flow', default=0.05, type=float,
    help='flow normalizing factor')
parser.add_argument('--alpha', default=0.9, type=float,
    help='alpha term for Adam')
parser.add_argument('--beta', default=0.999, type=float,
    help='beta term for Adam')
parser.add_argument('--weight_decay', default=4e-4, type=float,
    help='weight decay')
parser.add_argument('--lr_step_size', default=50, type=int,
    help='period of learning rate decay')
parser.add_argument('--lr_decay', default=0.1, type=float,
    help='factor of learning rate decay')
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
    input_transform = transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])
    target_transform = transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0,0],std=[args.div_flow,args.div_flow])
    ])
    co_transform = utils.Compose([
        utils.RandomCrop((320,448)),
        utils.RandomVerticalFlip(),
        utils.RandomHorizontalFlip()
    ])

    train_loader, valid_loader = KITTI_noc(dir=data_dir, batch_size=args.batch_size,
        input_transform=input_transform, target_transform=target_transform, co_transform=co_transform,
        split=0.9, num_workers=args.workers, pin_memory=pin_memory)

    print('--- Building model ---')
    cudnn.benchmark = True
    model = FlowNetS().to(device)
    model_name = 'FlowNetS'
    criterion = MultiScaleEPE(n_scales=args.n_scales, l_weight=args.l_weights)
    metric = EPE(div_flow=args.div_flow)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.alpha, args.beta), weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

    train = Train(model=model, data_loader=train_loader, optim=optimizer, criterion=criterion, metric=metric, device=device)
    train_writer = SummaryWriter(log_dir='./logs/train')
    valid = Test(model=model, data_loader=valid_loader, metric=metric, device=device)
    valid_writer = SummaryWriter(log_dir='./logs/valid')

    for epoch in range(1, args.n_epoches + 1):
        scheduler.step()
        train_loss, train_epe = train.run_epoch()
        print(">>>> Epoch {}: loss: {:.4f}, epe: {:.4f}".format(epoch, train_loss, train_epe))
        train_writer.add_scalar('avg loss', train_loss, epoch)
        train_writer.add_scalar('avg epe', train_epe, epoch)
        if epoch%10 == 0:
            valid_epe = valid.run_epoch()
            print(">>>> Validation: epe: {.4f}".format(valid_epe))
            valid_writer.add_scalar('avg epe', valid_epe, epoch)
            checkpoint(state={
                'name': model_name,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'epe': valid_epe,
                'div_flow': args.div_flow
            })
