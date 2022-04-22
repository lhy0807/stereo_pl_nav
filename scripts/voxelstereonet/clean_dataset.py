from __future__ import print_function, division
import os
import gc
import time
import argparse
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torchinfo import summary
import logging
import coloredlogs
from tqdm import tqdm

cudnn.benchmark = True
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=log)

THRESHOLD = 16

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='Voxel2D',
                    help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maximum disparity')
parser.add_argument('--dataset', required=True,
                    help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='base learning rate')
parser.add_argument('--lrepochs', type=str, required=True,
                    help='the epochs to decay lr: the downscale rate')
parser.add_argument('--batch_size', type=int, default=4,
                    help='training batch size')
parser.add_argument('--test_batch_size', type=int,
                    default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True,
                    help='number of epochs to train')
parser.add_argument('--logdir', required=True,
                    help='the directory to save logs and checkpoints')
parser.add_argument(
    '--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true',
                    help='continue training the model')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--summary_freq', type=int, default=100,
                    help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1,
                    help='the frequency of saving checkpoint')
parser.add_argument('--loader_workers', type=int, default=8,
                    help='Number of dataloader workers')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Choice of optimizer (adam or sgd)',
                    choices=["adam","sgd"])


# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def clean(config=None):
    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath, args.testlist, False)
    TrainImgLoader = DataLoader(
        train_dataset, 1, shuffle=False, num_workers=args.loader_workers, drop_last=False)
    TestImgLoader = DataLoader(
        test_dataset, 1, shuffle=False, num_workers=args.loader_workers, drop_last=False)

    # clean training set
    # train_total = len(TrainImgLoader)
    # train_valid = 0
    # q_bar = tqdm(TrainImgLoader)
    # for batch_idx, sample in enumerate(q_bar):
    #     vox_grid = sample["voxel_grid"][0].numpy()
    #     xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
    #     if xyz_v.shape[1] >= THRESHOLD:
    #         train_valid += 1
    #     q_bar.set_description_str(f"{train_valid} out of {batch_idx}: {train_valid / (batch_idx+1)}")
    #     q_bar.refresh()
    # print(f"Valid training sample rate {train_valid} out of {train_total}: {train_valid / train_total}")

    # clean training set
    test_total = len(TestImgLoader)
    test_valid = 0
    q_bar = tqdm(TestImgLoader)
    for batch_idx, sample in enumerate(q_bar):
        vox_grid = sample["voxel_grid"][0].numpy()
        xyz_v = np.asarray(np.where(vox_grid == 1)) # get back indexes of populated voxels
        if xyz_v.shape[1] >= THRESHOLD:
            test_valid += 1
        q_bar.set_description_str(f"{test_valid} out of {batch_idx}: {test_valid / (batch_idx+1)}")
        q_bar.refresh()
    print(f"Valid training sample rate {test_valid} out of {test_total}: {test_valid / test_total}")

if __name__ == "__main__":
    clean()