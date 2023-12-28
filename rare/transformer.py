"""
    Spatial Transformer Network: Localization_network + Grid_generator + Sampler
    The STN transforms an input image to a rectified image with a predicted TPS transformation
"""

from __future__ import print_function
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import rare.TPSGrid as tps


class STN(nn.Module):
    def __init__(self, channel_size, imgH, imgW, span_H, span_W, grid_H, grid_W):
        super(STN, self).__init__()
        """create base fiducial points (C')"""
        r1 = span_H
        r2 = span_W
        assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error
        y_coord = np.arange(-r1, r1+0.00001, 2.0*r1 / (grid_H-1))
        x_coord = np.arange(-r2, r2+0.00001, 2.0*r2 / (grid_W-1))
        base_fiducial_points = torch.Tensor(
            list(itertools.product(y_coord, x_coord)))
        Y, X = base_fiducial_points.split(1, dim=1)
        base_fiducial_points = torch.cat([X, Y], dim=1)
        self.base = base_fiducial_points

        """Localization network"""
        self.localization = nn.Sequential(
            nn.Conv2d(channel_size, 64, (3,3), (1,1), 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, (3, 3), (1, 1), 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, (3, 3), (1, 1), 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, (3, 3), (1, 1), 1), nn.ReLU(True), nn.MaxPool2d(2, 2))
        # Regressor for the grid_H*grid_W*2 coordinate matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512*2*17, 1024),
            nn.ReLU(True),
            nn.Linear(1024, grid_H*grid_W*2)) # the number of fiducial points K (= grid_H*grid_W)
        # Initialize the weights/bias with identity transformation
        bias = base_fiducial_points.view(-1)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(bias)

        """Grid generator"""
        self.tps = tps.TPSGridGen(imgH, imgW, base_fiducial_points)

    """Spatial transformer network forward function"""
    def forward(self, input):
        N = input.size(0)
        local = self.localization(input).view(N, -1) # [n, c*h*w]
        source_fiducial_point = torch.tanh(self.fc_loc(local)).view(N, -1, 2)
        grid = self.tps(source_fiducial_point) # Grid [n, h, w, 2]
        stn_output = F.grid_sample(input, grid, align_corners=True) # Sampler

        return stn_output # rectified images [n, c, h, w]