from __future__ import print_function, division
import os
import gc
import time
import argparse
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__, model_loss, calc_IoU
from utils import *
from torchinfo import summary
import logging
import coloredlogs
import wandb
from PIL import Image

cudnn.benchmark = True
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=log)

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
parser.add_argument('--lr', type=float, default=1e-3,
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
parser.add_argument('--loader_workers', type=int, default=4,
                    help='Number of dataloader workers')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='Choice of optimizer (adam or sgd)',
                    choices=["adam","sgd"])

parser.add_argument('--log_folder_suffix', type=str, default="")



# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def train(config=None):
    # train one sample
    def train_sample(sample, compute_metrics=False):
        model.train()

        imgL, imgR, voxel_gt = sample['left'], sample['right'], sample['voxel_grid']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        voxel_gt = voxel_gt.cuda()

        optimizer.zero_grad()

        voxel_ests = model(imgL, imgR)
        loss = model_loss(voxel_ests, voxel_gt)

        voxel_ests = voxel_ests[-1]
        scalar_outputs = {"loss": loss}
        img_outputs = {}
        voxel_outputs = []
        if compute_metrics:
            with torch.no_grad():
                voxel_outputs = [voxel_ests[0], voxel_gt[0]]
                IoU_list = []
                for idx, voxel_est in enumerate(voxel_ests):
                    IoU = calc_IoU(voxel_est, voxel_gt)
                    IoU_list.append(IoU.cpu().numpy())
                scalar_outputs["IoU"] = np.mean(IoU_list)

                left_filename = os.path.join(args.datapath, sample["left_filename"][0])
                left_img = Image.open(left_filename).convert('RGB')
                img_outputs["left_img"] = to_tensor(left_img)

        loss.backward()
        optimizer.step()

        return tensor2float(loss), tensor2float(scalar_outputs), voxel_outputs, img_outputs


    # test one sample
    @make_nograd_func
    def test_sample(sample, compute_metrics=True):
        model.eval()

        imgL, imgR, voxel_gt = sample['left'], sample['right'], sample['voxel_grid']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        voxel_gt = voxel_gt.cuda()

        voxel_ests = model(imgL, imgR)
        loss = model_loss(voxel_ests, voxel_gt)

        voxel_ests = voxel_ests[-1]
        scalar_outputs = {"loss": loss}
        img_outputs = {}
        voxel_outputs = [voxel_ests[0], voxel_gt[0]]
        IoU_list = []
        for idx, voxel_est in enumerate(voxel_ests):
            IoU = calc_IoU(voxel_est, voxel_gt)
            IoU_list.append(IoU.cpu().numpy())
        scalar_outputs["IoU"] = np.mean(IoU_list)

        left_filename = os.path.join(args.datapath, sample["left_filename"][0])
        left_img = Image.open(left_filename).convert('RGB')
        img_outputs["left_img"] = to_tensor(left_img)

        return tensor2float(loss), tensor2float(scalar_outputs), voxel_outputs, img_outputs

    # log inside wandb
    wandb.init(project="voxelDS", entity="nu-team", resume=True)
    # config = wandb.config
    log.info(f"wandb config: {config}")

    if args.model == 'MSNet2D':
        modelName = '2D-MobileStereoNet'
    elif args.model == 'MSNet3D':
        modelName = '3D-MobileStereoNet'
    elif args.model == "Voxel2D":
        modelName = '2D-MobileVoxelNet'

    print("==========================\n", modelName, "\n==========================")

    logdir_name = ""
    for k, v in config.items():
        logdir_name += str(k)
        logdir_name += '_'
        logdir_name += str(v)
        logdir_name += '_'

    if args.log_folder_suffix != "":
        logdir_name += args.log_folder_suffix
    
    args.logdir = os.path.join(args.logdir, logdir_name) + "/"

    log.info(f"Saving log at directory {args.logdir}")
    os.makedirs(args.logdir, mode=0o770, exist_ok=True)

    # create summary logger
    logger = SummaryWriter(args.logdir)

    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    train_dataset = StereoDataset(args.datapath, args.trainlist, True)
    test_dataset = StereoDataset(args.datapath, args.testlist, False)
    TrainImgLoader = DataLoader(
        train_dataset, config["batch_size"], shuffle=True, num_workers=args.loader_workers, drop_last=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    TestImgLoader = DataLoader(
        test_dataset, args.test_batch_size, shuffle=False, num_workers=args.loader_workers, drop_last=False, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    # model, optimizer
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()

    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    else:
        raise Exception("optimizer choice error!")

    # Transfer learning
    # model.module.load_mobile_stereo()

    # load parameters
    start_epoch = 0
    if args.resume:
        # find all checkpoints file and sort according to epoch id
        all_saved_ckpts = [fn for fn in os.listdir(
            args.logdir) if fn.endswith(".ckpt") and ("best" not in fn)]
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
        log.info("Loading the latest model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load the checkpoint file specified by args.loadckpt
        log.info("Loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'])
    log.info("Start at epoch {}".format(start_epoch))

    summary(model, [(2, 3, 400, 880), (2, 3, 400, 880)])

    best_checkpoint_loss = 100
    for epoch_idx in range(start_epoch, args.epochs):
        # lr_curr = adjust_learning_rate(optimizer, epoch_idx, config["lr"], args.lrepochs)
        # wandb.log({"lr_curr": lr_curr})

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, voxel_outputs, img_outputs = train_sample(
                sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_voxel(logger, 'train', voxel_outputs, global_step,
                           args.logdir, False)
                save_images(logger, "train", img_outputs, global_step)
                log.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, IoU = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                                         batch_idx,
                                                                                                         len(
                                                                                                             TrainImgLoader), loss,
                                                                                                         scalar_outputs["IoU"],
                                                                                                         time.time() - start_time))
                wandb.log({"train_IoU": scalar_outputs["IoU"], "train_loss": loss})
            else:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                log.info('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                           batch_idx,
                                                                                           len(
                                                                                               TrainImgLoader), loss,
                                                                                           time.time() - start_time))
            del scalar_outputs, voxel_outputs, img_outputs

            # if batch_idx >= 1000:
            #     break

        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
            ), 'optimizer': optimizer.state_dict()}
            torch.save(
                checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            test_loss, scalar_outputs, voxel_outputs, img_outputs = test_sample(
                sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_voxel(logger, 'test', voxel_outputs, global_step,
                           args.logdir, False)
                save_images(logger, "train", img_outputs, global_step)
                log.info('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, IoU = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                                        batch_idx,
                                                                                                        len(
                                                                                                            TestImgLoader), test_loss,
                                                                                                        scalar_outputs["IoU"],
                                                                                                        time.time() - start_time))
                wandb.log({"test_IoU": scalar_outputs["IoU"], "test_loss": test_loss})
            else:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                log.info('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                         batch_idx,
                                                                                         len(
                                                                                             TestImgLoader), test_loss,
                                                                                         time.time() - start_time))
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, voxel_outputs, img_outputs

        avg_test_scalars = avg_test_scalars.mean()

        save_scalars(logger, 'fulltest', avg_test_scalars,
                     len(TrainImgLoader) * (epoch_idx + 1))
        log.info(f"avg_test_scalars {avg_test_scalars}")
        wandb.log({"avg_test_loss": avg_test_scalars['loss']})

        # saving new best checkpoint
        if avg_test_scalars['loss'] < best_checkpoint_loss:
            best_checkpoint_loss = avg_test_scalars['loss']
            log.debug("Overwriting best checkpoint")
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(
            ), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/best.ckpt".format(args.logdir))

        gc.collect()


if __name__ == '__main__':
    # wandb.agent("lhy0807/stereo_pl_nav-scripts_voxelstereonet/iuzxah19", train)
    config = {"lr":args.lr, "batch_size":args.batch_size, "optimizer":"adam"}
    train(config=config)