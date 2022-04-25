# Copyright (c) 2021. All rights reserved.
from __future__ import print_function
import math
import torch.nn as nn
from torch import Tensor
import torch.utils.data
from collections import OrderedDict
from torch import reshape
import torch.nn.functional as F
from .submodule import feature_extraction, MobileV2_Residual, convbn, interweave_tensors, disparity_regression


class hourglass2D(nn.Module):
    def __init__(self, in_channels):
        super(hourglass2D, self).__init__()

        self.expanse_ratio = 2

        self.conv1 = MobileV2_Residual(
            in_channels, in_channels * 2, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv2 = MobileV2_Residual(
            in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv3 = MobileV2_Residual(
            in_channels * 2, in_channels * 4, stride=2, expanse_ratio=self.expanse_ratio)

        self.conv4 = MobileV2_Residual(
            in_channels * 4, in_channels * 4, stride=1, expanse_ratio=self.expanse_ratio)

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3,
                               padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels))

        self.redir1 = MobileV2_Residual(
            in_channels, in_channels, stride=1, expanse_ratio=self.expanse_ratio)
        self.redir2 = MobileV2_Residual(
            in_channels * 2, in_channels * 2, stride=1, expanse_ratio=self.expanse_ratio)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2))
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x))

        return conv6


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()
        # 48x128x240 => 64x64x128
        self.conv1 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                   nn.ReLU(inplace=True))

        # 64x64x128 => 128x16x32
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        # 128x16x32 => 256x4x8
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        self.linear1 = nn.Sequential(
            nn.Linear(256*3*7, 1024), nn.ReLU(inplace=True))
        # self.linear1 = nn.Sequential(
        #     nn.Linear(256*4*8, 1024), nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(inplace=True))

        # 256x1x1x1 => 256x2x2x2
        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(256, 128, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(0, 0, 0)),
                                     nn.BatchNorm3d(128),
                                     nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(128, 64, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(0, 0, 0)),
                                     nn.BatchNorm3d(64),
                                     nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(nn.ConvTranspose3d(64, 16, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(2, 2, 2)),
                                     nn.BatchNorm3d(16),
                                     nn.ReLU(inplace=True))
        self.deconv4 = nn.Sequential(nn.ConvTranspose3d(16, 1, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(2, 2, 2)),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape

        # encoding
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = reshape(conv3, (B, -1,))

        # latent
        linear1 = self.linear1(conv3)
        linear2 = self.linear2(linear1)
        latent = self.linear3(linear2)

        # decoding
        latent = reshape(latent, (B, 256, 1, 1, 1))

        deconv1 = self.deconv1(latent)
        deconv2 = self.deconv2(deconv1)
        deconv3 = self.deconv3(deconv2)
        out = self.deconv4(deconv3)

        out = torch.squeeze(out, 1)
        return out


class Voxel2D(nn.Module):
    def __init__(self, maxdisp):

        super(Voxel2D, self).__init__()

        self.maxdisp = maxdisp

        self.num_groups = 1

        self.volume_size = 24

        self.hg_size = 64

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(160, 128, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(128, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 16, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 32, kernel_size=(4, 3, 3), stride=[
                                              4, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.Conv3d(32, 16, kernel_size=(2, 3, 3), stride=[
                                              2, 1, 1], padding=[0, 1, 1]),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(16, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(nn.Conv2d(self.hg_size, self.hg_size, 1, 1, 0),
                                          nn.Sigmoid())

        self.encoder_decoder = UNet()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.kernel_size[2] * m.out_channels
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

        out = self.encoder_decoder(volume)
        return [out]
