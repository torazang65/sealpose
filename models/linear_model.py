from __future__ import absolute_import

import torch
import torch.nn as nn

'''
this folder and code is modified base on SemGCN code,
https://github.com/garyzhao/SemGCN
the Simple Yet Baseline model.
'''

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5, batch_norm=True):
        super(Linear, self).__init__()
        self.l_size = linear_size
        self.batch_norm = batch_norm

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.l_size)
            self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        if self.batch_norm:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        if self.batch_norm:
            y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, linear_size=1024, num_stage=2, p_dropout=0.5, num_joints=16, batch_norm=True):
        super(LinearModel, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.num_joints = num_joints
        self.batch_norm = batch_norm

        # 2d joints
        self.input_size = input_size  # 16 * 2
        # 3d joints
        self.output_size = output_size if output_size != 16 * 3 else 15 * 3  # 16 * 3

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        if batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout, batch_norm))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        """
        input: bx16x2 / bx32
        output: bx16x3
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], self.num_joints, 2)
        # pre-processing
        x = x.view(x.shape[0], self.num_joints * 2)  # 0924

        y = self.w1(x)
        if self.batch_norm:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        y = self.w2(y)

        # out: 15 joint ==> 16 joint
        if self.num_joints == 16:
            out = torch.cat([torch.zeros_like(y)[:,:3], y], 1).view(-1, self.num_joints, 3)  # Pad hip joint (0,0,0)
        else:
            out = y.view(-1, self.num_joints, 3)
        return out
