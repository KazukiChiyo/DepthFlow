import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from models.depthflownets import DepthFlowNetS
from backend import Train, Test, MultiScaleEPE, EPE, AdaBound
from datasets import KITTI_noc, utils
from tensorboardX import SummaryWriter
# from models.test import flownets_bn
from models import FlowNetS
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FlowNetS(in_channels=8, grouped=False, batch_norm=True).to(device)
summary(model, input_size=(8, 320, 448))
