import torch
import torch.nn as nn

from models.linear_model import Linear
from utils.loss import mpjpe


class LinearLossNet(nn.Module):
    def __init__(self, linear_size=1024, num_stage=2, p_dropout=0.5, num_joints=16, num_joints_partial=None, batch_norm=True):
        super(LinearLossNet, self).__init__()

        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.num_joints = num_joints
        
        self.batch_norm = batch_norm
        
        self.num_joints_partial = num_joints_partial if num_joints_partial is not None else num_joints

        # process input to linear size
        self.w1 = nn.Linear(num_joints * 2 + self.num_joints_partial * 3, self.linear_size)
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for l in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout, self.batch_norm))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x, y):
        if len(x.shape) == 2:
            x = x.view(x.shape[0], self.num_joints, 2)
        if len(y.shape) == 2:
            y = y.view(y.shape[0], self.num_joints_partial, 3)
        # pre-processing
        x = x.view(x.shape[0], self.num_joints * 2)  # 0924
        y = y.view(y.shape[0], self.num_joints_partial * 3)  # 0924
        
        x = torch.cat((x, y), dim=1)

        y = self.w1(x)
        if self.batch_norm:
            y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        out = self.w2(y)
        return out


class MarginBasedLoss:
    def __init__(self, margin_ratio=1, loss_type='mse'):
        self.margin_ratio = margin_ratio
        self.loss_type = loss_type

    def __call__(self, y_hat, label, energy_hat, energy_label):
        if self.loss_type == 'mse':
            delta = nn.MSELoss()(y_hat, label)
        elif self.loss_type == 'mpjpe':
            delta = mpjpe(y_hat, label)
        elif self.loss_type == 'l1':
            delta = nn.L1Loss()(y_hat, label)
        
        if self.margin_ratio < 0:
            margin_based_loss = - energy_hat + energy_label
        else:
            margin_based_loss = torch.max(
                self.margin_ratio * delta - energy_hat + energy_label,
                torch.zeros_like(delta),
            )
        return torch.mean(margin_based_loss)
    
    
    
class NCELoss:
    def __init__(self, temperature=1):
        self.temperature = temperature

    def __call__(self, y_hat, label, energy_hat, energy_label):
        max_energy = torch.max(energy_hat, energy_label)        
        nce_loss = - torch.log(
            1e-46 + torch.exp(torch.clamp((max_energy - energy_label) / self.temperature, max=50))
            / (torch.exp(torch.clamp((max_energy - energy_hat) / self.temperature, max=50))
               + torch.exp(torch.clamp((max_energy - energy_label) / self.temperature, max=50)))
        )
        
        # if torch.isnan(nce_loss).any():
        #     print('NCE Loss is nan')
        #     print(f"max_energy: {max_energy}")
        #     print(torch.exp((max_energy - energy_label)))
        #     print(torch.exp((max_energy - energy_hat)))
        #     print(nce_loss)
        return torch.mean(nce_loss)
    
    
# class NCELoss:
#     def __init__(self, temperature=1):
#         self.temperature = temperature

#     def __call__(self, y_hat, label, energy_hat, energy_label):
#         max_energy = torch.max(energy_hat, energy_label)
#         exp_label = torch.clamp(torch.exp((max_energy - energy_label) / self.temperature), max=1e10)
#         exp_hat = torch.clamp(torch.exp((max_energy - energy_hat) / self.temperature), max=1e10)
        
#         nce_loss = - torch.log(
#             1e-46 + exp_label / (exp_hat + exp_label)
#         )
#         # fix nan issue by clipping exp
#         nce_loss = torch.clamp(nce_loss, min=-1e10, max=1e10)

        
        
#         if torch.isnan(nce_loss).any():
#             print('NCE Loss is nan')
#             print(f"energy_hat: {energy_hat}")
#             print(f"energy_label: {energy_label}")
#             print(max_energy)
#             print(exp_label)
#             print(exp_hat)
#             print(nce_loss)
#         return torch.mean(nce_loss)