from __future__ import division
import os.path
import glob
from datasets.listdataset import ListDataset
import numpy as np
from datasets.util import *


try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
Dataset routines for KITTI_flow, 2015.
http://www.cvlibs.net/datasets/kitti/eval_flow.php
The dataset is not very big, you might want to only finetune on it for flownet
EPE are not representative in this dataset because of the sparsity of the GT.
OpenCV is needed to load 16bit png images
'''


def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(png_path, -1)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    return(flo_img)


def make_dataset(dPath, split, occ=True):
    """
    Will search in training folder for folders 'flow_noc' or 'flow_occ'
       and 'image_2' (KITTI 2015)
    """
    flow_dir = 'flow_occ' if occ else 'flow_noc'
    assert(os.path.isdir(os.path.join(dPath, flow_dir)))
    img_dir = 'image_2'
    assert(os.path.isdir(os.path.join(dPath, img_dir)))

    images = []
    for flow_map in glob.iglob(os.path.join(dPath, flow_dir, '*.png')):
        flow_map = os.path.basename(flow_map) # e.g. '000001_10.png'
        root_filename = flow_map[:-7] # '000001'
        flow_map = os.path.join(flow_dir, flow_map)
        img1 = os.path.join(img_dir, root_filename+'_10.png')
        img2 = os.path.join(img_dir, root_filename+'_11.png')
        if not (os.path.isfile(os.path.join(dPath, img1)) or os.path.isfile(os.path.join(dPath, img2))):
            continue
        images.append([[img1, img2], flow_map])

    return split2list(images, split)


def KITTI_loader(root,path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    return [cv2.imread(img)[:,:,::-1].astype(np.float32) for img in imgs],load_flow_from_png(flo)


def KITTI_occ(dPath, transform=None, target_transform=None,
              co_transform=None, split=None):
    """

    :param dPath: the path to the data folder, it must point to ./data_scene_flow/training
    :param transform:
    :param target_transform:
    :param co_transform:
    :param split: a float indicating train test split. If not provided, 0.9 is assumed
    :return: train dataset and test dataset
    """
    train_list, test_list = make_dataset(dPath, split, True)
    train_dataset = ListDataset(dPath, train_list, transform,
                                target_transform, co_transform,
                                loader=KITTI_loader)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(dPath, test_list, transform,
                               target_transform, CenterCrop((370,1224)),
                               loader=KITTI_loader)

    return train_dataset, test_dataset


def KITTI_noc(dPath, transform=None, target_transform=None,
              co_transform=None, split=None):
    """

    :param dPath: the path to the data folder, it must point to ./data_scene_flow/training
    :param transform:
    :param target_transform:
    :param co_transform:
    :param split: a float indicating train test split. If not provided, 0.9 is assumed
    :return: train dataset and test dataset
    """

    train_list, test_list = make_dataset(dPath, split, False)
    train_dataset = ListDataset(dPath, train_list, transform, target_transform, co_transform, loader=KITTI_loader)
    # All test sample are cropped to lowest possible size of KITTI images
    test_dataset = ListDataset(dPath, test_list, transform, target_transform, CenterCrop((370,1224)), loader=KITTI_loader)

    return train_dataset, test_dataset

# for single file testing
dataPath='/media/peixin/DATA/Research/DataSet/data_scene_flow/trainingSubset'
train,test=KITTI_noc(dataPath, split=0.9)
print(train.__getitem__(0))