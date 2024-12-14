#import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class mIOUMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.vals = {}
        self.counts = {}
        for i in range(self.n_classes):
            self.vals[i] = 0
            self.counts[i] = 0

    def update(self, val_d, count_d):
        miou = []
        for i in range(self.n_classes):
            self.vals[i] += val_d[i]
            self.counts[i] += count_d[i]
            if self.counts[i] > 0:
                miou.append(self.vals[i] / self.counts[i])

        self.avg = np.mean(miou)

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        
    def set_confusion_matrix(self, conf_mat):
        self.confusion_matrix = np.copy(conf_mat)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU
    
    def Mean_Intersection_over_Union_PerClass(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def jaccard(y_pred, y_true, num_classes):
    num_parts = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, 1)
    y_pred = y_pred.type(torch.LongTensor)
    y_true = y_true.type(torch.LongTensor)
    y_pred = F.one_hot(y_pred, num_classes=num_classes)
    y_true = F.one_hot(y_true, num_classes=num_classes)
    nbs = y_pred.shape[0]
    ious = []
    for nb in range(nbs):
        img_ious = []
        for i in range(num_parts):
            pred = y_pred[nb,:,:,i]
            gt = y_true[nb,:,:,i]
            inter = torch.logical_and(pred, gt)
            union = torch.logical_or(pred, gt)
            iou = torch.sum(inter, [0,1]) / torch.sum(union, [0,1])
            if torch.sum(gt, [0,1]) > 0:
                img_ious.append(iou)
        img_ious = torch.stack(img_ious)
        ious.append(torch.mean(img_ious))

    ious = torch.stack(ious)
    legal_labels = ~torch.isnan(ious)
    ious = ious[legal_labels]
    return torch.mean(ious)


def jaccard_perpart(y_pred, y_true, num_classes):
    num_parts = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, 1)
    y_pred = y_pred.type(torch.LongTensor)
    y_true = y_true.type(torch.LongTensor)
    y_pred = F.one_hot(y_pred, num_classes=num_classes)
    y_true = F.one_hot(y_true, num_classes=num_classes)
    nbs = y_pred.shape[0]
    ious = {}
    counts = {}
    for i in range(num_parts):
        pred = y_pred[:,:,:,i]
        gt = y_true[:,:,:,i]
        inter = torch.logical_and(pred, gt)
        union = torch.logical_or(pred, gt)
        iou = torch.sum(inter, [1,2]) / torch.sum(union, [1,2])
        legal = torch.sum(gt, [1,2]) > 0
        ious[i] = torch.sum(iou[legal])
        counts[i] = torch.sum(legal)

    return ious, counts


