# Copyright (c) 2021. All rights reserved.
from __future__ import print_function
import math
import torch.nn as nn
from torch import Tensor
import torch.utils.data
from collections import OrderedDict
from torch import reshape
import torch.nn.functional as F
from .submodule import MobileV1_Residual, MobileV2_Residual, convbn, interweave_tensors, groupwise_correlation

class feature_extraction(nn.Module):
    def __init__(self, add_relus=False):
        super(feature_extraction, self).__init__()

        self.expanse_ratio = 3
        self.inplanes = 16
        if add_relus:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 16, 4, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(16, 16, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True),
                                           MobileV2_Residual(16, 16, 1, self.expanse_ratio),
                                           nn.ReLU(inplace=True)
                                           )
        else:
            self.firstconv = nn.Sequential(MobileV2_Residual(3, 16, 4, self.expanse_ratio),
                                           MobileV2_Residual(16, 16, 1, self.expanse_ratio),
                                           MobileV2_Residual(16, 16, 1, self.expanse_ratio)
                                           )

        self.layer1 = self._make_layer(MobileV1_Residual, 16, 3, 1, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 16, 8, 2, 1, 1)
        self.layer3 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 1)
        self.layer4 = self._make_layer(MobileV1_Residual, 32, 3, 1, 1, 2)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        feature_volume = torch.cat((l2, l3, l4), dim=1)

        return feature_volume

class UNet(nn.Module):
    def __init__(self, cost_vol_type) -> None:
        super(UNet, self).__init__()
        # 48x128x240 => 64x64x128
        if cost_vol_type == "full" or cost_vol_type == "gwc":
            self.conv1 = nn.Sequential(nn.Conv2d(48, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))
        elif cost_vol_type == "voxel" or cost_vol_type == "eveneven" or cost_vol_type == "gwcvoxel":
            self.conv1 = nn.Sequential(nn.Conv2d(17, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(24, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 10)),
                                    nn.ReLU(inplace=True))

        # 64x64x128 => 128x16x32
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        # 128x16x32 => 256x4x8
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0)),
                                   nn.ReLU(inplace=True))

        self.linear1 = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True))

        # 256x1x1x1 => 256x2x2x2
        self.deconv1 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                                     nn.BatchNorm3d(32),
                                     nn.ReLU(inplace=True))

        self.deconv2 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(0, 0, 0), bias=False),
                                     nn.BatchNorm3d(16),
                                     nn.ReLU(inplace=True))

        self.deconv3 = nn.Sequential(nn.ConvTranspose3d(16, 4, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(2, 2, 2), bias=False),
                                     nn.BatchNorm3d(4),
                                     nn.ReLU(inplace=True))
        self.deconv4 = nn.Sequential(nn.ConvTranspose3d(4, 1, kernel_size=(6, 6, 6), stride=(2, 2, 2), padding=(2, 2, 2)),
                                     nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape

        # encoding
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = reshape(conv3, (B, -1,))

        # latent
        latent = self.linear1(conv3)

        # decoding
        latent = reshape(latent, (B, 64, 1, 1, 1))

        deconv1 = self.deconv1(latent)
        deconv2 = self.deconv2(deconv1)
        deconv3 = self.deconv3(deconv2)
        out = self.deconv4(deconv3)

        out = torch.squeeze(out, 1)
        return out


class Voxel2D(nn.Module):
    def __init__(self, maxdisp, cost_vol_type="even"):

        super(Voxel2D, self).__init__()

        self.maxdisp = maxdisp
        self.cost_vol_type = cost_vol_type

        self.num_groups = 1

        self.volume_size = 12

        self.hg_size = 64

        self.dres_expanse_ratio = 3

        self.feature_extraction = feature_extraction(add_relus=True)

        self.preconv11 = nn.Sequential(convbn(80, 64, 1, 1, 0, 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 32, 1, 1, 0, 1))

        self.conv3d = nn.Sequential(nn.Conv3d(1, 8, kernel_size=(8, 3, 3), stride=[8, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(8),
                                    nn.ReLU(),
                                    nn.Conv3d(8, 16, kernel_size=(4, 3, 3), stride=[
                                              4, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.Conv3d(16, 8, kernel_size=(2, 3, 3), stride=[
                                              2, 1, 1], padding=[0, 1, 1], bias=False),
                                    nn.BatchNorm3d(8),
                                    nn.ReLU())

        self.volume11 = nn.Sequential(convbn(8, 1, 1, 1, 0, 1),
                                      nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(nn.Conv2d(self.hg_size, self.hg_size, 1, 1, 0),
                                          nn.Sigmoid())

        self.encoder_decoder = UNet(self.cost_vol_type)

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
        
        if self.cost_vol_type == "gwc" or self.cost_vol_type == "gwcvoxel":
            self.gwc_conv3d = nn.Sequential(nn.Conv3d(40, 20, 1, 1), nn.Conv3d(20,1,1,1))

    def forward(self, L, R, voxel_cost_vol=[0]):
        features_L = self.feature_extraction(L)
        features_R = self.feature_extraction(R)

        if self.cost_vol_type != "gwc" and self.cost_vol_type != "gwcvoxel":
            featL = self.preconv11(features_L)
            featR = self.preconv11(features_R)
        else:
            featL = features_L
            featR = features_R

        B, C, H, W = featL.shape

        # default even = 24
        iter_size = self.volume_size
        
        if self.cost_vol_type == "full":
            # full disparity = 24x2 = 48
            iter_size = int(self.volume_size*2)
        elif self.cost_vol_type == "eveneven":
            # eveneven = 24/2 = 12
            iter_size = int(self.volume_size/2)
        elif self.cost_vol_type == "voxel" or self.cost_vol_type == "gwcvoxel":
            # voxel  = 16+1 = 17
            iter_size = len(voxel_cost_vol) + 1

        volume = featL.new_zeros([B, self.num_groups, iter_size, H, W])
        
        if self.cost_vol_type == "gwc":
            volume = featL.new_zeros([B, 40, 24, H, W])
        if self.cost_vol_type == "gwcvoxel":
            volume = featL.new_zeros([B, 40, 17, H, W])

        for i in range(iter_size):
            if i > 0:
                if self.cost_vol_type == "even":
                    j = 2*i
                elif self.cost_vol_type == "eveneven":
                    j = 4*i
                elif self.cost_vol_type == "front":
                    j = int(i + self.volume_size*2)
                elif self.cost_vol_type == "back":
                    j = i
                elif self.cost_vol_type == "full":
                    j = i
                elif self.cost_vol_type == "voxel":
                    j = int(voxel_cost_vol[i-1][0])
                elif self.cost_vol_type == "gwc":
                    volume[:, :, i, :, i:] = groupwise_correlation(featL[:, :, :, i:], featR[:, :, :, :-i], 40)
                    continue
                elif self.cost_vol_type == "gwcvoxel":
                    j = int(voxel_cost_vol[i-1][0])
                    volume[:, :, i, :, j:] = groupwise_correlation(featL[:, :, :, j:], featR[:, :, :, :-j], 40)
                    continue
                x = interweave_tensors(featL[:, :, :, j:], featR[:, :, :, :-j])
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, j:] = x
            else:
                if self.cost_vol_type == "gwc" or self.cost_vol_type == "gwcvoxel":
                    volume[:, :, i, :, :] = groupwise_correlation(featL, featR, 40)
                    continue

                x = interweave_tensors(featL, featR)
                x = torch.unsqueeze(x, 1)
                x = self.conv3d(x)
                x = torch.squeeze(x, 2)
                x = self.volume11(x)
                volume[:, :, i, :, :] = x

        volume = volume.contiguous()
        volume = torch.squeeze(volume, 1)

        if self.cost_vol_type == "gwc" or self.cost_vol_type == "gwcvoxel":
            volume = self.gwc_conv3d(volume)
            volume = torch.squeeze(volume, 1)

        out = self.encoder_decoder(volume)
        return [out]