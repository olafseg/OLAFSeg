import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_dilated_mask(image, class_index, dilation_kernel, ignore_label=None):
    # Mask of the specific class
    mask = torch.eq(image, class_index).float()

    if ignore_label is not None:
        mask_ignore = 1.0 - torch.eq(image, ignore_label).float()
        mask = tf.multiply(mask_ignore, mask)

    dilated_mask = nn.MaxPool2d(kernel_size=dilation_kernel, stride=1, padding=1)(mask)

    return dilated_mask

def compute_adj_mat(image, adj_mat, num_classes, present_classes, ignore_label, dilation_kernel, weighted):

    num_present_classes = present_classes.shape[0]
    i = 1

    while (i < num_present_classes):
        j = i + 1

        first_dilated_mask = compute_dilated_mask(image, present_classes[i], dilation_kernel)

        while (j < num_present_classes):
            second_dilated_mask = compute_dilated_mask(image, present_classes[j], dilation_kernel)

            intersection = torch.mul(first_dilated_mask, second_dilated_mask)

            adjacent_pixels = torch.sum(intersection).type(torch.int)

            # WeightedAdjMat - The class1-class2 value contains the number of adjacent pixels if the 2 classes
            # are adjacent,  0 otherwise
            if weighted:
                indices = torch.Tensor([[present_classes[i]], [present_classes[j]], [0]])
                values = torch.reshape(adjacent_pixels, [1]).cpu()
                shape = [num_classes, num_classes, 1]
                delta = torch.sparse_coo_tensor(indices, values, shape)
                adj_mat = adj_mat + delta.to_dense()

            # SimpleAdjMat - The class1-class2 value contains 1 if the 2 classes are adjacent, 0 otherwise
            else:
                value = adjacent_pixels > 0
                value = value.float()
                indices = torch.Tensor([[present_classes[i], present_classes[j], 0]])
                values = torch.reshape(value, [1])
                shape = [num_classes, num_classes, 1]
                delta = torch.sparse_coo_tensor(indices, values, shape)
                adj_mat = adj_mat + delta.to_dense()

            j = j + 1

        
        i = i + 1

    return adj_mat

def adjacent_graph_loss(pred, gt, num_classes, weighted=True,
                        ignore_label=None, lambda_loss=0.1,
                        dilation_kernel=3):
    pred = F.interpolate(pred, size=gt.shape[1:], mode='bilinear', align_corners=False)
    pred = torch.argmax(pred, dim=1)
    
    concat = torch.cat([torch.reshape(pred, [-1]), torch.reshape(gt, [-1])], 0)
    unique = torch.unique(concat, sorted=True)
    
    logits_adj_mat = torch.zeros([num_classes, num_classes, 1], dtype=torch.int32)
    labels_adj_mat = torch.zeros([num_classes, num_classes, 1], dtype=torch.int32)
    
    logits_adj_mat = compute_adj_mat(image=pred,
                                     adj_mat=logits_adj_mat,
                                     num_classes=num_classes,
                                     present_classes=unique,
                                     ignore_label=ignore_label,
                                     dilation_kernel=dilation_kernel,
                                     weighted=weighted)

    labels_adj_mat = compute_adj_mat(image=gt,
                                     adj_mat=labels_adj_mat,
                                     num_classes=num_classes,
                                     present_classes=unique,
                                     ignore_label=ignore_label,
                                     dilation_kernel=dilation_kernel,
                                     weighted=weighted)
    
    logits_adj_mat = logits_adj_mat.type(torch.DoubleTensor)
    labels_adj_mat = labels_adj_mat.type(torch.DoubleTensor)
    if weighted:
        logits_adj_mat = F.normalize(logits_adj_mat, dim=0)
        labels_adj_mat = F.normalize(labels_adj_mat, dim=0)
        
    loss = nn.MSELoss()(logits_adj_mat, labels_adj_mat)
    return loss * lambda_loss

def objmask_loss(pred, macro_gt, num_classes, weighted=True,
                 ignore_label=None, lambda_loss=0.001,
                 dilation_kernel=3, label_weights=None):
    pred = F.interpolate(pred, size=macro_gt.shape[1:], mode='bilinear', align_corners=False)

    macro_class_logits = torch.split(pred, [1, num_classes-1], dim=1)
    macro_logits_sum = []
    for i in range(len(macro_class_logits)):
            macro_logits_sum.append(torch.sum(macro_class_logits[i], axis=1))
    
    macro_pred = torch.stack(macro_logits_sum, axis=1)
    loss = nn.CrossEntropyLoss(weight=label_weights)(macro_pred, macro_gt)
    return loss * lambda_loss

def crossentropy_loss(pred, gt, lambda_loss=1.0, label_weights=None):
    pred = F.interpolate(pred, size=gt.shape[1:], mode='bilinear', align_corners=False)

    loss = nn.CrossEntropyLoss(weight=label_weights)(pred, gt)
    return loss * lambda_loss