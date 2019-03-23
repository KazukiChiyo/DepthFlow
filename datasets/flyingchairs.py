import os.path
import glob
from datasets.listdataset import ListDataset
from datasets.util import split2list


def make_dataset(dPath, split=None):
    """
    Each example contain three files:
    '[name]_img1.ppm  [name]_img2.ppm  [name]_flow.flo'
    :return: train list and test list with strings of file names in each of the list
    """

    images = []
    for flow_map in sorted(glob.glob(os.path.join(dPath,'*_flow.flo'))):
        # os.path.basename(path) function returns the tail of the path.
        # E.g.: The basename of '/foo/bar/item' returns 'item'
        flow_map = os.path.basename(flow_map) # get string such as '00001_flow.flo'
        root_filename = flow_map[:-9] # get string such as '00001'
        img1 = root_filename+'_img1.ppm' # get '00001_img1'
        img2 = root_filename+'_img2.ppm' # get '00001_img2'
        if not (os.path.isfile(os.path.join(dPath,img1)) and os.path.isfile(os.path.join(dPath,img2))):
            continue

        images.append([[img1,img2],flow_map]) # these are all strings

    return split2list(images, split)


def flying_chairs(dPath, transform=None, target_transform=None,
                  co_transform=None, split=None):
    """

    :param dPath: the path to the data folder, it must point to ./FlyingChairs_release/data
    :param transform:
    :param target_transform:
    :param co_transform:
    :param split: a float indicating train test split. If not provided, 0.9 is assumed
    :return: train dataset and test dataset
    """
    train_list, test_list = make_dataset(dPath,split)
    train_dataset = ListDataset(dPath, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(dPath, test_list, transform, target_transform)

    return train_dataset, test_dataset
