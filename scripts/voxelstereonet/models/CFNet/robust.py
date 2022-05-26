from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
from datasets import listfiles as ls
from datasets import eth3dLoader as DA
from datasets import MiddleburyLoader as mid
import gc
import wandb
cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Cascade and Fused Cost Volume for Robust Stereo Matching(CFNet)')
parser.add_argument('--model', default='cfnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset',  help='dataset name', choices=__datasets__.keys(), default="ds")
parser.add_argument('--datapath',  help='data path', default="/home/chris/pl_ws/src/stereo_pl_nav/datasets/DS")
parser.add_argument('--trainlist', help='training list', default="/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_train.txt")
parser.add_argument('--testlist', help='testing list', default="/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/filenames/DS_test.txt")

parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default="/home/chris/pl_ws/src/stereo_pl_nav/scripts/voxelstereonet/models/CFNet/", help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
# logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

wandb.init(project="DSFineTune", entity="nu-team")

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    bestepoch = 0
    error = 100
    bestepochkitti = 0
    kittierror = 100
    bestepocheth3d = 0
    eth3derror = 100
    bestepochmid = 0
    miderror = 100
    for epoch_idx in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                wandb.log({"train_loss":loss, "train_EPE":scalar_outputs["EPE"], "train_D1":scalar_outputs["D1"]})
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    disp_ests = model(imgL, imgR)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, dataset = 'kitti', compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    toppad, rightpad = sample['top_pad'], sample['right_pad']
    # print(toppad)
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    disp_ests, pred3_s3, pred3_s4  = model(imgL, imgR)
    if dataset == 'mid':
        # print(disp_gt.size())
        #mask = (disp_gt < args.maxdisp * 2) & (disp_gt > 0)
        mask = disp_gt > 0
        disp_ests[0] = disp_ests[0][:, toppad:, :-rightpad]
        pred3_s3[0] = pred3_s3[0][:, toppad:, :-rightpad]
        pred3_s4[0] = pred3_s4[0][:, toppad:, :-rightpad]

        disp_ests = F.upsample(disp_ests[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear', align_corners=True).squeeze(1)
        pred3_s3 = F.upsample(pred3_s3[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear',
                               align_corners=True).squeeze(1)
        pred3_s4 = F.upsample(pred3_s4[0].unsqueeze(1) * 2, [disp_gt.size()[1], disp_gt.size()[2]], mode='bilinear',
                              align_corners=True).squeeze(1)

        disp_ests = [disp_ests]
        pred3_s3 = [pred3_s3]
        pred3_s4 = [pred3_s4]
        inrange_s4 = (disp_gt > 0) & (disp_gt < 256 * 2) & mask

    # mask = disp_gt > 0
    else:
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        inrange_s4 = (disp_gt > 0) & (disp_gt < 256) & mask
    loss = model_loss(disp_ests, disp_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"disp_est": disp_ests, "disp_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1_preds3"] = [D1_metric(pred, disp_gt, mask) for pred in pred3_s3]
    scalar_outputs["D1_preds4"] = [D1_metric(pred, disp_gt, mask) for pred in pred3_s4]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres1s3"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in pred3_s3]
    scalar_outputs["Thres1s4"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in pred3_s4]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres2s3"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in pred3_s3]
    scalar_outputs["Thres2s4"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in pred3_s4]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func.apply(disp_est, disp_gt) for disp_est in disp_ests]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
