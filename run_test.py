import argparse
from path import Path
import torch
import torch.nn.functional as F
from models.depthflownets import DepthFlowNetS
from models import FlowNetS
import models
from tqdm import tqdm

import torchvision.transforms as transforms
from imageio import imread, imwrite
import cv2
import numpy as np

from datasets import utils


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', default='/home/peixin/Documents/Vogel/KITTI/trainingSubset',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model', default='/home/peixin/Documents/Vogel/ckpts/FlowNetS_depth:adam_epoch_500_epe9.6607_df20.0000.pth')
parser.add_argument('--output', '-o', metavar='DIR', default='/home/peixin/Documents/Vogel/flow',
                    help='path to output folder. If not set, will be created in data folder')

parser.add_argument('--div-flow', default=20, type=float,
                    help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", metavar='EXT', default=['png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float,
                    help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default='bilinear', help='if not set, will output FlowNet raw input,'
                    'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--depth', '-d', type=bool,default=True, help='test with depth information')



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()

@torch.no_grad()
def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('save everything to {}'.format(save_path))
    save_path.makedirs_p()
    # Data loading code
    img_dir=data_dir / 'image_2'

    img_pairs = []
    for ext in args.img_exts:
        test_files = img_dir.files('*0.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + '1.{}'.format(ext))
            if args.depth:
                depth_dir = data_dir / 'train_disparity'
                dep0=depth_dir / (file.namebase[:-1] + '0.{}'.format(ext))
                dep1=depth_dir / (file.namebase[:-1] + '1.{}'.format(ext))
                img_pairs.append([file, img_pair, dep0, dep1])
            else:
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    # load pretrained weight and create model
    network_data = torch.load(args.pretrained)
    print("using pre-trained model")
    #model = models.__dict__[network_data['name']](network_data).to(device)
    if args.depth:
        numChannels=8
        isGrouped=True
    else:
        numChannels=6
        isGrouped = False
    model=FlowNetS(in_channels=numChannels, grouped=isGrouped, batch_norm=True)
    model.load_state_dict(network_data['state_dict'])
    model.to(device)


    model.eval()

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']

    # define transformations
    input_transform = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    # depth transform will be ignored if arg.depth=False
    depth_transform = transforms.Compose([
        utils.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[255]),
        transforms.Normalize(mean=[0.5], std=[1])
    ])

    if args.depth:
        for (img1_file, img2_file,dep1,dep2) in tqdm(img_pairs):
            # apply transformations
            img1 = input_transform(imread(img1_file))
            img2 = input_transform(imread(img2_file))
            depth1=depth_transform(np.expand_dims(cv2.imread(dep1,0),axis=2))
            depth2=depth_transform(np.expand_dims(cv2.imread(dep2,0),axis=2))
            input_var = torch.cat((img1, img2, depth1,depth2)).unsqueeze(0)
            input_var = input_var.to(device)
            # compute output
            computeOutput(model,input_var,img1,img1_file)
    else:
        for (img1_file, img2_file) in tqdm(img_pairs):
            # apply transformations
            img1 = input_transform(imread(img1_file))
            img2 = input_transform(imread(img2_file))
            input_var = torch.cat((img1, img2)).unsqueeze(0)
            input_var = input_var.to(device)
            # compute output
            computeOutput(model,input_var,img1,img1_file)


def computeOutput(model,input_var,img1,img1_file):
    output = model(input_var)
    if args.upsampling is not None:
        output = F.interpolate(output, size=img1.size()[-2:], mode=args.upsampling, align_corners=False)
    for suffix, flow_output in zip(['flow', 'inv_flow'], output):
        filename = save_path / '{}{}'.format(img1_file.namebase[:-1], suffix)
        rgb_flow = flow2rgb(args.div_flow * flow_output, max_value=args.max_flow)
        to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
        imwrite(filename + '.png', to_save)
        # Make the flow map a HxWx2 array as in .flo files
        to_save = flow_output.cpu().numpy().transpose(1, 2, 0)
        np.save(filename + '.npy', to_save)


if __name__ == '__main__':
    main()