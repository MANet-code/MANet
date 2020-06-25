import torch
import config
import torchvision
import torch.nn as nn
import models

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

#多视角中的分组以及计算权重

#from .Model import Model, init_weights
#import models.InceptionV4 as Icpv4


class OneConvFc(nn.Module):
    """
    1*1 conv + fc to obtain the grouping schema
    """
    def __init__(self):
        super(OneConvFc, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_features=27*27, out_features=1)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GroupSchema(nn.Module):
    """
    differences from paper:
    1. Considering the amount of params, we use 1*1 conv instead of  fc
    2. if the scores are all very small, it will cause a big problem in params' update,
    so we add a softmax layer to normalize the scores after the convolution layer
    """
    #根据视角的自身区分度计算分数return self.sft(view_scores)Bx12
    def __init__(self):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc()
        self.sft = nn.Softmax(dim=1)

    def forward(self, raw_view):
        """
        :param raw_view: [V N C H W]
        :return:
        """
        scores = []
        for batch_view in raw_view:
            # batch_view: [N C H W]
            # y: [N]
            y = self.score_layer(batch_view)
            y = torch.sigmoid(torch.log(torch.abs(y)))
            scores.append(y)
        # 最终的view_scores: [N V]
        view_scores = torch.stack(scores, dim=0).transpose(0, 1)
        #注意这一步dim=0时，则就是讲list[scores]中的12个矩阵变成12维，按照顺序，第i维是第i个矩阵，size=(i，B，1)
        #print('view_scores1',view_scores.shape) view_scores1 torch.Size([43/42, 12, 1])

        view_scores = view_scores.squeeze(dim=-1)
        #print('view_scores2',view_scores.shape) view_scores2 torch.Size([43/42, 12])

        #print('view_scores3',self.sft(view_scores).shape) view_scores3 torch.Size([43, 12])
        #现在就可以理解为12代表每一个视角的分数
        return self.sft(view_scores)




def view_pool(ungrp_views, view_scores, num_grps=7):
    """
    :param ungrp_views: [V C H W]
    :param view_scores: [V]
    :param num_grps the num of groups. used to calc the interval of each group.
    :return: grp descriptors [(grp_descriptor, weight)]
    """
    #可以理解为在进行组内视角池化，具体怎么把每个视角分成组，并且组数的问题都需要再看一下，
    #最后会得到每个组的描述符带着每个组的权重，这里希望能改一下，权重用点云来计算


    #这个函数最后得到的分数到底有啥用哈？？？
    def calc_scores(scores):
        """
        :param scores: [score1, score2 ....]
        :return:
        """
        n = len(scores)
        #print('scores',scores)
        #print('len(scores)',n=12)
        s = torch.ceil(scores[0]*n)
        #print('scores[0]*n取天井',s)
        for idx, score in enumerate(scores):
            if idx == 0:
                continue
            s += torch.ceil(score*n)
            #print('score*n取天井',s)
        s /= n
        #print('final_s',s)
        return s

    interval = 1 / (num_grps + 1)
    # begin = 0
    view_grps = [[] for i in range(num_grps)]
    score_grps = [[] for i in range(num_grps)]

    for idx, (view, view_score) in enumerate(zip(ungrp_views, view_scores)):
        begin = 0
        for j in range(num_grps):
            right = begin + interval
            if j == num_grps-1:
                right = 1.1
            if begin <= view_score < right:
                view_grps[j].append(view)
                score_grps[j].append(view_score)
                #这里把组的分数改成点云计算的权重！！！！
            begin += interval
    #print('score_grps:',score_grps)会得到很多组tensor0.0839
    #这一步？？？
    view_grps = [sum(views)/len(views) for views in view_grps if len(views) > 0]
    #print('len(view_grps)(观察这个数值与len(scores)不相等)',len(view_grps)=1)
        
    score_grps = [calc_scores(scores) for scores in score_grps if len(scores) > 0]
    #print('len(score_grps)(观察这个数值与len(scores),len(view_grps)是否相等)',len(score_grps)=1)
    shape_des = map(lambda a, b: a*b, view_grps, score_grps)
    
    shape_des = sum(shape_des)/sum(score_grps)
    #print('len(shape_des2)',len(shape_des)=256)
    # !!! if all scores are very small, it will cause some problems in params' update
    if sum(score_grps) < 0.1:
        # shape_des = sum(view_grps)/len(score_grps)
        print(sum(score_grps), score_grps)
    # print('score total', score_grps)
    return shape_des

    
def group_pool(final_view, scores):
    """
    view pooling + group fusion
    :param final_view: # [N V C H W]
    :param scores: [N V] scores
    :return: shape descriptor
    """
    shape_descriptors = []

    for idx, (ungrp_views, view_scores) in enumerate(zip(final_view, scores)):
        # ungrp_views: [V C H W]
        # view_scores: [V]

        # view pooling
        shape_descriptors.append(view_pool(ungrp_views, view_scores))
        #print('len(shape_descriptors)',len(shape_descriptors))
    # [N C H W]
    y = torch.stack(shape_descriptors, 0)
    #print('y.size', y.size()([43/42, 256, 6, 6]))
    return y




    
class BaseFeatureNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, pretrained=True):
        super(BaseFeatureNet, self).__init__()
        base_model_name = base_model_name.upper()
        self.fc_features = None
        self.num_views = 12
        #关于多视角分组的相关内容
        self.group_schema = GroupSchema()
        self.avg_pool_2 = nn.AvgPool2d(8,count_include_pad = False)
        
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
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
            base_model_list = list(model.children())[0:13]
            base_model_list.extend(list(model.children())[14:17])
            self.features = nn.Sequential(*base_model_list)
            self.feature_len = 2048


        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

#这里如果能使用4张GPU，则bachsize=32,如果3张，则bachsize=43，42，43
    def forward(self, x):
        # x = x[:,0]
        # if len(x.size()) == 5:
        batch_sz = x.size(0)
        #print('batchsize',batch_sz)
        view_num = x.size(1)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        
        #x.shape torch.Size([384, 3, 224, 224])
        
        with torch.no_grad():
            x = self.features[:1](x)
            
            #x.shape torch.Size([384, 64, 55, 55]
            #x = self.cbam(x)
        x = self.features[1:3](x)
        #print('x_12.shape',x.shape)[64,27,27]
        

        
        #加上根据区分度计算如何分组的代码
        
        y1 = x.view(batch_sz, self.num_views, x.shape[-3], x.shape[-2], x.shape[-1])
        #print('y1.size[32/43/42,12,64,27,27]', y1.size())

        raw_view = y1.transpose(0, 1)
        #print('raw_view.size[12, 32/43/42, 64, 27, 27]', raw_view.size())

        # [N V] scores
        view_scores = self.group_schema(raw_view)
        # print('[32 12]', view_scores.size())
        
        # [NV 256 6 6]
        x = self.features[3:](x)
        #print('x_3.shape[384/516/504, 256, 6, 6]',x.shape)
        

        final_view = x.view(batch_sz,self.num_views, x.shape[-3],x.shape[-2], x.shape[-1])
        #print('final_view.size[32/43/42 12 256 6 6]', final_view.size())

        shape_decriptors = group_pool(final_view, view_scores)
        #print('shape_decriptors',shape_decriptors.shape,shape_decriptors torch.Size([43, 256, 6, 6]))
        #x = self.avg_pool_2(shape_decriptors)
        #print('x',x.shapex torch.Size([43, 256, 1, 1]) )
        x = x.view(x.size(0), -1)
        #print('x.shape',x.shape([43, 256]))
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


