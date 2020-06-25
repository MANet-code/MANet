from models import *

class GAPNet(nn.Module):
    def __init__(self, n_neighbor=20, num_classes=20):
        super(GAPNet, self).__init__()
        self.n_neighbor = n_neighbor
        self.trans_net = transform_net(19,16,3)
        self.gap = gap_layer(self.n_neighbor)

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
        locals_max_transform, _  = torch.max(torch.cat(local_features_1, dim = 1),dim = -1,keepdim=True)
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

        locals_max, _  = torch.max(torch.cat(local_features, dim = 1),dim = -1,keepdim=True) 
        neighbors_features = torch.cat(attns, dim = -1).permute(0,3,1,2)
        
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
