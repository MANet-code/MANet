from models import *
import torch.nn as nn
import torch.nn.functional as F



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
class GAPNet(nn.Module):
    def __init__(self, n_neighbor=20, num_classes=20, gate_channels1 = 16, gate_channels2 = 64 ):
        super(GAPNet, self).__init__()
        self.n_neighbor = n_neighbor
        self.gate_channels1 = gate_channels1
        self.gate_channels2 = gate_channels2
        self.trans_net = transform_net(19,16,3)
        self.gap = gap_layer(self.n_neighbor)
        self.cbam1 = CBAM(self.gate_channels1)
        self.cbam2 = CBAM(self.gate_channels2)
        self.conv2d1 = conv_2d(67, 64, 1)
        self.conv2d2 = conv_2d(64, 64, 1)
        self.conv2d3 = conv_2d(64, 64, 1)
        self.conv2d4 = conv_2d(64, 128, 1)
        self.conv2d5 = conv_2d(384, 1024, 1)
        self.mlp1 = nn.Sequential(
            fc_layer(1024, 512, True),
            nn.Dropout(p=0.5)
        )
        self.mlp2 = nn.Sequential(
            fc_layer(512, 256, True),
            nn.Dropout(p=0.5)
        )
        self.mlp3 = nn.Linear(256, num_classes)

    def forward(self, x):
        n_heads_1 = 1
        attns_1 = []
        local_features_1 = []
        
        edge_feature_1,  locals_1 = self.gap(x,self.n_neighbor)
        attns_1.append(edge_feature_1)
        local_features_1.append(locals_1)
        
        neighbors_features_1 = torch.cat(attns_1, dim = -1).permute(0,3,1,2)

        neighbors_features_1 = torch.cat((x,neighbors_features_1),dim = 1)
        locals_max_transform = torch.cat(local_features_1, dim = 1)
        locals_max_transform = self.cbam1(locals_max_transform)
        locals_max_transform, _  = torch.max(locals_max_transform,dim = -1,keepdim=True)
        x_trans = self.trans_net(neighbors_features_1,locals_max_transform)
        x1 = x.squeeze(-1).transpose(2, 1)
        x1= torch.bmm(x1, x_trans)
        x = x1.transpose(2, 1)
        
        n_heads = 4
        attns = []
        local_features = []
        for i in range(n_heads):
            edge_feature,local = self.gap(x,self.n_neighbor)
            attns.append(edge_feature)
            local_features.append(local)

        locals_max = torch.cat(local_features, dim = 1)
        # locals_max.shape(32,64,1024,20)
        neighbors_features = torch.cat(attns, dim = -1).permute(0,3,1,2)
        locals_max = self.cbam2(locals_max)
        
        locals_max, _= torch.max(locals_max,dim = -1,keepdim=True)
        x1 = torch.cat((torch.unsqueeze(x,dim =-1),neighbors_features),dim = 1)
        
        x1 = self.conv2d1(x1)
        
        x2 = self.conv2d2(x1)
        
        x3 = self.conv2d3(x2)
        
        x4 = self.conv2d4(x3)

        x5 = torch.cat((x1, x2, x3, x4,locals_max), dim=1)
        x5 = self.conv2d5(x5)
        x5, _ = torch.max(x5, dim=-2, keepdim=True)

        net = x5.view(x5.size(0), -1)
        net = self.mlp1(net)
        net = self.mlp2(net)
        net = self.mlp3(net)

        return net
