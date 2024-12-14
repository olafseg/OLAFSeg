#import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class low_level_feature_extractor(nn.Module):
    def __init__(self, backbone, BatchNorm):
        super(low_level_feature_extractor, self).__init__()
        if backbone == 'resnet101' or backbone == 'resnet50' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        
        
        self.conv1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
                                       #nn.Dropout(0.5))
                                       
                                       
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())
                                       #nn.Dropout(0.5))
                                       
        self.conv_cat = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False),
                                       BatchNorm(512),
                                       nn.ReLU())
                                                                      

        self._init_weight()


    def forward(self, x2, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        x2 = self.conv2(x2)
        
        low_level_feat = F.interpolate(low_level_feat, size=x2.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x2, low_level_feat), dim=1)
        x = self.conv_cat(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_low_level_feature_extractor(backbone, BatchNorm):
    return low_level_feature_extractor(backbone, BatchNorm)                