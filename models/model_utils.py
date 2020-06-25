import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import config
import torch.nn.functional as F
from models import *


ALEXNET = "ALEXNET"
DENSENET121 = "DENSENET121"
VGG13 = "VGG13"
VGG13BN = "VGG13BN"
VGG11BN = 'VGG11BN'
RESNET50 = "RESNET50"
RESNET101 = "RESNET101"
INCEPTION_V3 = 'INVEPTION_V3'
INCEPTION_V4 = 'INVEPTION_V4'
# MVCNN functions
class conv_2d_nobias(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(conv_2d_nobias, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(conv_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(fc_layer, self).__init__()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.fc(x)
        return x



class transform_net(nn.Module):
    def __init__(self, in_ch, locals_max_transform, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.locals_max_transform = locals_max_transform
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(144, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1024, 1))
        self.fc1 = fc_layer(1024, 512, bn=True)
        self.fc2 = fc_layer(512, 256, bn=True)
        self.fc3 = nn.Linear(256, K*K)

    
    def forward(self, x, locals_max_transform):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        
        x = torch.cat((x,locals_max_transform),dim = 1)
        
        x = self.conv2d3(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(3).view(1,9).repeat(x.size(0),1)
        iden = iden.to(device=config.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

    
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

def get_neighbors(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    
    return feature
   
    
class gap_layer(nn.Module):
    def __init__(self, n_neighbor):
        super(gap_layer, self).__init__()
        self.n_neighbor = n_neighbor
        
        self.conv2d1 = conv_2d_nobias(3,16,1)
        self.conv2d2 = conv_2d(3,16,1)
        self.conv2d3 = conv_2d(16,1,1)
        
        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.ELU()
    def forward(self, x, n_neighbor):

        batch_size = x.size()[0]
        num_dim = x.size()[1]
        num_point = x.size()[2]
        if len(x.size()) == 3:
            x = x.unsqueeze(3)
        neighbors = get_neighbors(x)
        x = x.permute(0, 2, 3, 1).repeat(1, 1, self.n_neighbor, 1)
        x = (x - neighbors).permute(0, 3, 1, 2)
        
        new_feature = self.conv2d1(x)
        self_attention = self.conv2d3(new_feature)
        
        edge_feature = self.conv2d2(x)
        x1 = edge_feature.permute(0,2,3,1)
        
        neibor_attention = self.conv2d3(edge_feature)
        logits = (self_attention + neibor_attention).permute(0,2,1,3)
        coefs = self.act1(logits)
        coefs = F.softmax(coefs,dim = -1)
        x2 = coefs
        
        vals = torch.matmul(x2, x1)
        ret = self.act2(vals)
        
        return ret,  edge_feature

