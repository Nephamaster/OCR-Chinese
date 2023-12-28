"""
    The Grid generator estimates the TPS(thin-plate-spline) transformation parameters,
    and generates a sampling grid
"""

import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_distance(input_p, target_p):
    N = input_p.size(0)
    M = target_p.size(0)
    pairwise_diff = input_p.view(N, 1, 2) - target_p.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class TPSGridGen(nn.Module):

    def __init__(self, target_H, target_W, base_fiducial_points):
        super(TPSGridGen, self).__init__()
        self.H = target_H
        self.W = target_W
        # base fiducial points (C') must be [K, 2]
        assert base_fiducial_points.ndimension() == 2
        assert base_fiducial_points.size(1) == 2
        K = base_fiducial_points.size(0)
        self.num_points = K # constant even number
        base_fiducial_points = base_fiducial_points.float()

        # create padded kernel matrix (delta C') [K+3, K+3]
        kernel = torch.zeros(K+3, K+3)
        R = compute_distance(base_fiducial_points, base_fiducial_points)
        kernel[:K, :K].copy_(R)
        kernel[:K, -3].fill_(1)
        kernel[-3, :K].fill_(1)
        kernel[:K, -2:].copy_(base_fiducial_points)
        kernel[-2:, :K].copy_(base_fiducial_points.transpose(0, 1))
        # compute inverse matrix (delta C'^-1)
        inverse_kernel = torch.inverse(kernel)

        # create target coordinate matrix (p')
        HW = target_H * target_W
        target_coord = list(itertools.product(range(target_H), range(target_W)))
        target_coord = torch.Tensor(target_coord) # HW x 2
        Y, X = target_coord.split(1, dim = 1)
        Y = Y * 2 / (target_H - 1) - 1
        X = X * 2 / (target_W - 1) - 1
        target_coord = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        R_ = compute_distance(target_coord, base_fiducial_points)
        target_coord = torch.cat([
            R_, torch.ones(HW, 1), target_coord], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate', target_coord)

    def forward(self, source_fiducial_points):
        # C must be [N, K, 2]
        assert source_fiducial_points.ndimension() == 3
        assert source_fiducial_points.size(1) == self.num_points
        assert source_fiducial_points.size(2) == 2
        batch_size = source_fiducial_points.size(0)

        Y = torch.cat([source_fiducial_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        TPS_matrix = torch.matmul(Variable(self.inverse_kernel), Y) # T
        source_coordinate = torch.matmul(Variable(self.target_coordinate), TPS_matrix) # p = T*p'
        return source_coordinate.view(batch_size, self.H, self.W, 2)