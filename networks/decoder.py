#import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'resnet50' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        self.conv_y_aspp = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                       BatchNorm(48),
                                       nn.ReLU())
                                       
                               
                                       
        self.last_conv1 = nn.Sequential(nn.Conv2d(304+48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
                                       
                                       
                                       
        self.last_conv2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))  
                                       
        self.last_conv3 = nn.Sequential(nn.Conv2d(256, num_classes, kernel_size=1, stride=1))  
                                                        
        self._init_weight()
        


    def forward(self, x, low_level_feat, y_aspp):
    
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        y_aspp = self.conv_y_aspp(y_aspp)
        y_aspp = F.interpolate(y_aspp, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x, y_aspp, low_level_feat), dim=1)
        x = self.last_conv1(x)
        x = self.last_conv2(x)
        x = self.last_conv3(x)
        

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

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)                