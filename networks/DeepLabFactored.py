#import necessary modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.decoder import build_decoder
from networks.low_level_feature_extractor import build_low_level_feature_extractor
from networks.backbone import build_backbone 



class DeepLabFactored(nn.Module):
    def __init__(self, num_anim_classes, num_inanim_classes, backbone='resnet101', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(DeepLabFactored, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.features = build_low_level_feature_extractor(backbone, BatchNorm)

        self.anim_aspp_low = build_aspp('drn', output_stride, BatchNorm)
        self.anim_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.anim_decoder = build_decoder(num_anim_classes, backbone, BatchNorm)

        self.inanim_aspp_low = build_aspp('drn', output_stride, BatchNorm)
        self.inanim_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.inanim_decoder = build_decoder(num_inanim_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x4, _, x2, low_level_feat = self.backbone(input)
        
        features = self.features(x2, low_level_feat)

        anim_low_aspp = self.anim_aspp_low(features)
        anim_x = self.anim_aspp(x4)
        anim_x = self.anim_decoder(anim_x, low_level_feat, anim_low_aspp)
        anim_x = F.interpolate(anim_x, size=input.size()[2:], mode='bilinear', align_corners=True)

        inanim_low_aspp = self.inanim_aspp_low(features)
        inanim_x = self.inanim_aspp(x4)
        inanim_x = self.inanim_decoder(inanim_x, low_level_feat, inanim_low_aspp)
        inanim_x = F.interpolate(inanim_x, size=input.size()[2:], mode='bilinear', align_corners=True)
       
        return anim_x, inanim_x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.features, self.anim_aspp, self.inanim_aspp, self.anim_aspp_low, self.inanim_aspp_low, self.anim_decoder, self.inanim_decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p