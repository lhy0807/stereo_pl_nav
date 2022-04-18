# Copyright (c) 2021. All rights reserved.
from __future__ import print_function
import math
from pyexpat import model
import torch.nn as nn
from torch import Tensor
import torch.utils.data
from collections import OrderedDict
import torch.nn.functional as F
from .submodule import feature_extraction, MobileV2_Residual, convbn, interweave_tensors, disparity_regression


class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2))
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x))

        return conv6


class Voxel2D(nn.Module):
    def __init__(self, maxdisp):

        super(Voxel2D, self).__init__()

        self.maxdisp = maxdisp

        self.num_groups = 1

        self.volume_size = 48

        self.hg_size = 48

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(320, 256, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(256, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.voxel_grid = nn.Upsample(size=(24,48), mode="bilinear")

        self.dres0 = nn.Sequential(MobileV2_Residual(self.volume_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                   nn.ReLU(inplace=True),
                                   MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio))

        self.encoder_decoder1 = hourglass2D(self.hg_size)

        self.encoder_decoder2 = hourglass2D(self.hg_size)

        self.encoder_decoder3 = hourglass2D(self.hg_size)

        self.output_layer = nn.Sequential(MobileV2_Residual(self.hg_size, self.hg_size, 1, self.dres_expanse_ratio),
                                            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def select_layers(self, name):
        selected_layer_names = ["feature_extraction",
                                "preconv11", "conv3d", "volume11"]
        for layer in selected_layer_names:
            if layer in name:
                return True
        return False

    def load_mobile_stereo(self, mobile_stereo_model="../models/MSNet2D_SF_DS_KITTI2015.ckpt"):
        print("Loading from MobileStereoNet 2D")
        state_dict = torch.load(mobile_stereo_model)["model"]
        model_dict = self.state_dict()
        pretrained_dict = OrderedDict()
        for name, param in state_dict.items():
            name = name.split("module.")[1]
            if name not in model_dict:
                continue
            if not self.select_layers(name):
                continue
            pretrained_dict[name] = param

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        del state_dict

        for param in self.feature_extraction.parameters():
            param.requires_grad = False

        for param in self.preconv11.parameters():
            param.requires_grad = False

        for param in self.conv3d.parameters():
            param.requires_grad = False

        for param in self.volume11.parameters():
            param.requires_grad = False

    def forward(self, L, R):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)

        featL = self.preconv11(features_L)
        featR = self.preconv11(features_R)

        B, C, H, W = featL.shape
        volume = featL.new_zeros([B, self.num_groups, self.volume_size, H, W])
        for i in range(self.volume_size):
            if i > 0:
                x = interweave_tensors(featL[:, :, :, i:], featR[:, :, :, :-i])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, i:] = x
            else:
                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        vox = self.voxel_grid(volume)
        vox = torch.transpose(vox, 1, -1)

        cost0 = self.dres0(vox)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.encoder_decoder1(cost0)  # [2, hg_size, 64, 128]
        out2 = self.encoder_decoder2(out1)
        out3 = self.encoder_decoder3(out2)

        if self.training:
            return [self.output_layer(out1), self.output_layer(out2), self.output_layer(out3)]

        else:
            return [self.output_layer(out3)]
