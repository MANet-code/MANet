import torch
import config
import torchvision
import torch.nn as nn
import models
import pretrainedmodels
import pretrainedmodels.utils as utils
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models import *




class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(
            2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(
            gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    
class BaseFeatureNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, pretrained=True):
        super(BaseFeatureNet, self).__init__()
        base_model_name = base_model_name.upper()
        self.fc_features = None
        self.gate_channels = 64
        #self.cbam = CBAM(self.gate_channels)

        if base_model_name == models.VGG13:
            base_model = torchvision.models.vgg13(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.VGG11BN:
            base_model = torchvision.models.vgg11_bn(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.VGG13BN:
            base_model = torchvision.models.vgg13_bn(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.ALEXNET:
            # base_model = torchvision.models.alexnet(pretrained=pretrained)
            base_model = torchvision.models.alexnet(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.RESNET50:
            base_model = torchvision.models.resnet50(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif base_model_name == models.RESNET101:
            base_model = torchvision.models.resnet101(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])

        elif base_model_name == models.INCEPTION_V3:
            base_model = torchvision.models.inception_v3(pretrained=pretrained)
            base_model_list = list(base_model.children())[0:13]
            base_model_list.extend(list(base_model.children())[14:17])
            self.features = nn.Sequential(*base_model_list)
            self.feature_len = 2048
        elif base_model_name == models.INCEPTION_V4:
            model_name = 'inceptionv4' # could be fbresnet152 or inceptionresnetv2
            model = pretrainedmodels.__dict__[model_name](num_classes=40, pretrained=True)
            base_model_list = list(model.children())[0:13]
            base_model_list.extend(list(model.children())[14:17])
            self.features = nn.Sequential(*base_model_list)
            self.feature_len = 2048

        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

    def forward(self, x):
        # x = x[:,0]
        # if len(x.size()) == 5:
        batch_sz = x.size(0)
        view_num = x.size(1)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        #x.shape torch.Size([384, 3, 224, 224])
        
        with torch.no_grad():
            x = self.features[:1](x)
            
            #x.shape torch.Size([384, 64, 55, 55]
            #x = self.cbam(x)
        x = self.features[1:](x)
        print(x.shape)
        #x.shape torch.Size([384, 256, 6, 6])
        x = x.view(x.size(0), -1)
        x = self.fc_features(x) if self.fc_features is not None else x
        #x.shape torch.Size([384, 4096])
        # max view pooling
        x_view = x.view(batch_sz, view_num, -1)
        x, _ = torch.max(x_view, 1)

        return x, x_view


class BaseClassifierNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, num_classes=40, pretrained=True):
        super(BaseClassifierNet, self).__init__()
        base_model_name = base_model_name.upper()
        if base_model_name in (models.VGG13, models.VGG13BN, models.ALEXNET, models.VGG11BN):
            self.feature_len = 4096
        elif base_model_name in (models.RESNET50, models.RESNET101, models.INCEPTION_V3, models.INCEPTION_V4):
            self.feature_len = 2048
        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

        self.classifier = nn.Linear(self.feature_len, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x


class MVCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(MVCNN, self).__init__()
        base_model_name = config.base_model_name
        num_classes = config.view_net.num_classes
        print(f'\ninit {base_model_name} model...\n')
        self.features = BaseFeatureNet(base_model_name, pretrained)
        self.classifier = BaseClassifierNet(base_model_name, num_classes, pretrained)

    def forward(self, x):
        x, _ = self.features(x)
        x = self.classifier(x)
        return x


