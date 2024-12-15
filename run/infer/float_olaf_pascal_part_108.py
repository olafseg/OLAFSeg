import glob 
import math
import sys
import os
import cv2
import glob
import numpy as np
import pickle
import matplotlib.pylab as plt
import time
import random
import math
import collections
import pickle
import queue
import collections
import threading
import functools
from tqdm import tqdm
from typing import Dict, Type, Any, Callable, Union, List, Optional


import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torch.utils.model_zoo as model_zoo
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast



class FutureResult(object):
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._result = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def put(self, result):
        with self._lock:
            assert self._result is None, 'Previous result has\'t been fetched.'
            self._result = result
            self._cond.notify()

    def get(self):
        with self._lock:
            if self._result is None:
                self._cond.wait()

            res = self._result
            self._result = None
            return res


_MasterRegistry = collections.namedtuple('MasterRegistry', ['result'])
_SlavePipeBase = collections.namedtuple('_SlavePipeBase', ['identifier', 'queue', 'result'])


class SlavePipe(_SlavePipeBase):
    """Pipe for master-slave communication."""

    def run_slave(self, msg):
        self.queue.put((self.identifier, msg))
        ret = self.result.get()
        self.queue.put(True)
        return ret


class SyncMaster(object):
    """An abstract `SyncMaster` object.
    - During the replication, as the data parallel will trigger an callback of each module, all slave devices should
    call `register(id)` and obtain an `SlavePipe` to communicate with the master.
    - During the forward pass, master device invokes `run_master`, all messages from slave devices will be collected,
    and passed to a registered callback.
    - After receiving the messages, the master device should gather the information and determine to message passed
    back to each slave devices.
    """

    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self._master_callback = master_callback
        self._queue = queue.Queue()
        self._registry = collections.OrderedDict()
        self._activated = False

    def __getstate__(self):
        return {'master_callback': self._master_callback}

    def __setstate__(self, state):
        self.__init__(state['master_callback'])

    def register_slave(self, identifier):
        """
        Register an slave device.
        Args:
            identifier: an identifier, usually is the device id.
        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        if self._activated:
            assert self._queue.empty(), 'Queue is not clean before next initialization.'
            self._activated = False
            self._registry.clear()
        future = FutureResult()
        self._registry[identifier] = _MasterRegistry(future)
        return SlavePipe(identifier, self._queue, future)

    def run_master(self, master_msg):
        """
        Main entry for the master device in each forward pass.
        The messages were first collected from each devices (including the master device), and then
        an callback will be invoked to compute the message to be sent back to each devices
        (including the master device).
        Args:
            master_msg: the message that the master want to send to itself. This will be placed as the first
            message when calling `master_callback`. For detailed usage, see `_SynchronizedBatchNorm` for an example.
        Returns: the message to be sent back to the master device.
        """
        self._activated = True

        intermediates = [(0, master_msg)]
        for i in range(self.nr_slaves):
            intermediates.append(self._queue.get())

        results = self._master_callback(intermediates)
        assert results[0][0] == 0, 'The first result should belongs to the master.'

        for i, res in results:
            if i == 0:
                continue
            self._registry[i].result.put(res)

        for i in range(self.nr_slaves):
            assert self._queue.get() is True

        return results[0][1]

    @property
    def nr_slaves(self):
        return len(self._registry)



def _sum_ft(tensor):
    """sum over the first and last dimention"""
    return tensor.sum(dim=0).sum(dim=-1)


def _unsqueeze_ft(tensor):
    """add new dementions at the front and the tail"""
    return tensor.unsqueeze(0).unsqueeze(-1)


_ChildMessage = collections.namedtuple('_ChildMessage', ['sum', 'ssum', 'sum_size'])
_MasterMessage = collections.namedtuple('_MasterMessage', ['sum', 'inv_std'])


class _SynchronizedBatchNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_SynchronizedBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

        self._sync_master = SyncMaster(self._data_parallel_master)

        self._is_parallel = False
        self._parallel_id = None
        self._slave_pipe = None

    def forward(self, input):
        # If it is not parallel computation or is in evaluation mode, use PyTorch's implementation.
        if not (self._is_parallel and self.training):
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training, self.momentum, self.eps)

        # Resize the input to (B, C, -1).
        input_shape = input.size()
        input = input.view(input.size(0), self.num_features, -1)

        # Compute the sum and square-sum.
        sum_size = input.size(0) * input.size(2)
        input_sum = _sum_ft(input)
        input_ssum = _sum_ft(input ** 2)

        # Reduce-and-broadcast the statistics.
        if self._parallel_id == 0:
            mean, inv_std = self._sync_master.run_master(_ChildMessage(input_sum, input_ssum, sum_size))
        else:
            mean, inv_std = self._slave_pipe.run_slave(_ChildMessage(input_sum, input_ssum, sum_size))

        # Compute the output.
        if self.affine:
            # MJY:: Fuse the multiplication for speed.
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std * self.weight) + _unsqueeze_ft(self.bias)
        else:
            output = (input - _unsqueeze_ft(mean)) * _unsqueeze_ft(inv_std)

        # Reshape it.
        return output.view(input_shape)

    def __data_parallel_replicate__(self, ctx, copy_id):
        self._is_parallel = True
        self._parallel_id = copy_id

        # parallel_id == 0 means master device.
        if self._parallel_id == 0:
            ctx.sync_master = self._sync_master
        else:
            self._slave_pipe = ctx.sync_master.register_slave(copy_id)

    def _data_parallel_master(self, intermediates):
        """Reduce the sum and square-sum, compute the statistics, and broadcast it."""

        # Always using same "device order" makes the ReduceAdd operation faster.
        # Thanks to:: Tete Xiao (http://tetexiao.com/)
        intermediates = sorted(intermediates, key=lambda i: i[1].sum.get_device())

        to_reduce = [i[1][:2] for i in intermediates]
        to_reduce = [j for i in to_reduce for j in i]  # flatten
        target_gpus = [i[1].sum.get_device() for i in intermediates]

        sum_size = sum([i[1].sum_size for i in intermediates])
        sum_, ssum = ReduceAddCoalesced.apply(target_gpus[0], 2, *to_reduce)
        mean, inv_std = self._compute_mean_std(sum_, ssum, sum_size)

        broadcasted = Broadcast.apply(target_gpus, mean, inv_std)

        outputs = []
        for i, rec in enumerate(intermediates):
            outputs.append((rec[0], _MasterMessage(*broadcasted[i * 2:i * 2 + 2])))

        return outputs

    def _compute_mean_std(self, sum_, ssum, size):
        """Compute the mean and standard-deviation with sum and square-sum. This method
        also maintains the moving average on the master device."""
        assert size > 1, 'BatchNorm computes unbiased standard-deviation, which requires size > 1.'
        mean = sum_ / size
        sumvar = ssum - sum_ * mean
        unbias_var = sumvar / (size - 1)
        bias_var = sumvar / size

        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.data

        return mean, bias_var.clamp(self.eps) ** -0.5



class SynchronizedBatchNorm2d(_SynchronizedBatchNorm):
    r"""Applies Batch Normalization over a 4d input that is seen as a mini-batch
    of 3d inputs
    .. math::
        y = \frac{x - mean[x]}{ \sqrt{Var[x] + \epsilon}} * gamma + beta
    This module differs from the built-in PyTorch BatchNorm2d as the mean and
    standard-deviation are reduced across all devices during training.
    For example, when one uses `nn.DataParallel` to wrap the network during
    training, PyTorch's implementation normalize the tensor on each device using
    the statistics only on that device, which accelerated the computation and
    is also easy to implement, but the statistics might be inaccurate.
    Instead, in this synchronized version, the statistics will be computed
    over all training samples distributed on multiple devices.
    Note that, for one-GPU or CPU-only case, this module behaves exactly same
    as the built-in PyTorch implementation.
    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and gamma and beta are learnable parameter vectors
    of size C (where C is the input size).
    During training, this layer keeps a running estimate of its computed mean
    and variance. The running sum is kept with a default momentum of 0.1.
    During evaluation, this running mean/variance is used for normalization.
    Because the BatchNorm is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial BatchNorm
    Args:
        num_features: num_features from an expected input of
            size batch_size x num_features x height x width
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Default: 0.1
        affine: a boolean value that when set to ``True``, gives the layer learnable
            affine parameters. Default: ``True``
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    Examples:
        >>> # With Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = SynchronizedBatchNorm2d(100, affine=False)
        >>> input = torch.autograd.Variable(torch.randn(20, 100, 35, 45))
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(SynchronizedBatchNorm2d, self)._check_input_dim(input)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4, x3, x2, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k == 'conv1.weight' or k == 'bn1.running_mean' or k == 'bn1.running_var' or k == 'bn1.weight' or k == 'bn1.bias':
                    continue
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class ResNet_obj(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True):
        self.inplanes = 64
        super(ResNet_obj, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict, strict = False) 

def ResNet101_obj(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_obj(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet101(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model

def ResNet50(output_stride, BatchNorm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained)
    return model



def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm)
    else:
        raise NotImplementedError

def build_backbone_obj(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return ResNet101_obj(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm)
    else:
        raise NotImplementedError 

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

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

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)



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
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

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




class DeepLab(nn.Module):
    def __init__(self, backbone='resnet101', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone_obj(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

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
        modules = [self.aspp, self.decoder]
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



class Decoder_Factored(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_Factored, self).__init__()
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

def build_decoder_factored(num_classes, backbone, BatchNorm):
    return Decoder_Factored(num_classes, backbone, BatchNorm)

###low level feature extractor ####################

class low_level_feature_extractor(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
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

def build_low_level_feature_extractor(num_classes, backbone, BatchNorm):
    return low_level_feature_extractor(num_classes, backbone, BatchNorm)
    
    



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

        self.features = build_low_level_feature_extractor(num_anim_classes, backbone, BatchNorm)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        self.anim_aspp_low = build_aspp('drn', output_stride, BatchNorm)
        self.anim_aspp = build_aspp(backbone, output_stride, BatchNorm)
        
        self.anim_decoder = build_decoder_factored(num_anim_classes, backbone, BatchNorm)

        self.inanim_aspp_low = build_aspp('drn', output_stride, BatchNorm)
        self.inanim_aspp = build_aspp(backbone, output_stride, BatchNorm)
        
        self.inanim_decoder = build_decoder_factored(num_inanim_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x4, x3, x2, low_level_feat = self.backbone(input)
        features = self.features(x2, low_level_feat)
        
        anim_x_low_aspp = self.anim_aspp_low(features)
        anim_x = self.anim_aspp(x4)
        anim_x = self.anim_decoder(anim_x, low_level_feat, anim_x_low_aspp)
        anim_x = F.interpolate(anim_x, size=input.size()[2:], mode='bilinear', align_corners=True)

        inanim_x_low_aspp = self.inanim_aspp_low(features)
        inanim_x = self.inanim_aspp(x4)
        inanim_x = self.inanim_decoder(inanim_x, low_level_feat, inanim_x_low_aspp)
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
        modules = [self.features, self.anim_aspp, self.inanim_aspp, self.anim_decoder, self.inanim_decoder, self.anim_aspp_low, self.inanim_aspp_low]
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



class CallbackContext(object):
    pass

class DataParallelWithCallback(DataParallel):
    """
    Data Parallel with a replication callback.
    An replication callback `__data_parallel_replicate__` of each module will be invoked after being created by
    original `replicate` function.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
        # sync_bn.__data_parallel_replicate__ will be invoked.
    """

    def replicate(self, module, device_ids):
        modules = super(DataParallelWithCallback, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created by original replication.
    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`
    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.
    We guarantee that the callback on the master copy (the first copy) will be called ahead of calling the callback
    of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.
    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate



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
        self.sqiou = miou



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



class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f' % (epoch, lr))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10



def part_obj_to_datasetclass(obj_classes=21, animate=True):
    
    # Animate parts : Head(1), Torso(2), Leg(3), Tail(4), Wing(5), Arm(6), Neck(7)
    # Animate parts : Eye(8), Ear(9), Nose(10), Muzzle(11), Horn(12), Mouth(13), Hair(14), Foot(15),
    #                 Hand(16), Paw(17), Hoof(18), Beak(19)
    map_pc = {}
    for i in range(obj_classes):
        map_pc[i] = {}
        
    # Animate objects

    map_pc[3][1] = 10  # Bird
    map_pc[3][19] = 11
    map_pc[3][2] = 12
    map_pc[3][7] = 13
    map_pc[3][5] = 14
    map_pc[3][3] = 15
    map_pc[3][15] = 16
    map_pc[3][4] = 17

    map_pc[8][1] = 36  # Cat
    map_pc[8][8] = 37
    map_pc[8][9] = 38
    map_pc[8][10] = 39
    map_pc[8][2] = 40
    map_pc[8][7] = 41
    map_pc[8][3] = 42
    map_pc[8][17] = 43
    map_pc[8][4] = 44

    map_pc[10][1] = 46  # Cow
    map_pc[10][9] = 47
    map_pc[10][11] = 48
    map_pc[10][12] = 49
    map_pc[10][2] = 50
    map_pc[10][7] = 51
    map_pc[10][3] = 52
    map_pc[10][4] = 53

    map_pc[12][1] = 55  # Dog
    map_pc[12][8] = 56
    map_pc[12][9] = 57
    map_pc[12][10] = 58
    map_pc[12][2] = 59
    map_pc[12][7] = 60
    map_pc[12][3] = 61
    map_pc[12][17] = 62
    map_pc[12][4] = 63
    map_pc[12][11] = 64

    map_pc[13][1] = 65  # Horse
    map_pc[13][9] = 66
    map_pc[13][11] = 67
    map_pc[13][2] = 68
    map_pc[13][7] = 69
    map_pc[13][3] = 70
    map_pc[13][4] = 71
    map_pc[13][18] = 72

    map_pc[15][1] = 77  # Person
    map_pc[15][8] = 78
    map_pc[15][9] = 79
    map_pc[15][10] = 80
    map_pc[15][13] = 81
    map_pc[15][14] = 82
    map_pc[15][2] = 83
    map_pc[15][7] = 84
    map_pc[15][6] = 85
    map_pc[15][16] = 86
    map_pc[15][3] = 87
    map_pc[15][15] = 88

    map_pc[17][1] = 91  # Sheep
    map_pc[17][9] = 92
    map_pc[17][11] = 93
    map_pc[17][12] = 94
    map_pc[17][2] = 95
    map_pc[17][7] = 96
    map_pc[17][3] = 97
    map_pc[17][4] = 98
    
    
    map_pc[1][1] = 1  # Aeroplane
    map_pc[1][4] = 2
    map_pc[1][3] = 3
    map_pc[1][5] = 4
    map_pc[1][2] = 5

    map_pc[2][2] = 6  # Bicycle
    map_pc[2][14] = 7
    map_pc[2][15] = 8
    map_pc[2][16] = 9
    
    map_pc[4][0] = 18  # Boat

    map_pc[5][12] = 19 # Bottle
    map_pc[5][13] = 20

    map_pc[6][17] = 21  # Bus
    map_pc[6][18] = 22
    map_pc[6][19] = 23
    map_pc[6][7] = 24
    map_pc[6][20] = 25
    map_pc[6][2] = 26
    map_pc[6][6] = 27
    map_pc[6][11] = 28

    map_pc[7][17] = 29  # Car
    map_pc[7][18] = 30
    map_pc[7][7] = 31
    map_pc[7][20] = 32
    map_pc[7][2] = 33
    map_pc[7][6] = 34
    map_pc[7][11] = 35
    
    map_pc[9][0] = 45  # Chair
    
    map_pc[11][0] = 54  # Dining Table

    map_pc[14][2] = 73  # Motorbike
    map_pc[14][15] = 74
    map_pc[14][14] = 75
    map_pc[14][6] = 76

    map_pc[16][9] = 89  # Potted plant
    map_pc[16][10] = 90
    
    map_pc[18][0] = 99  # Sofa

    map_pc[19][21] = 100 # Train
    map_pc[19][22] = 101
    map_pc[19][23] = 102
    map_pc[19][6] = 103
    map_pc[19][24] = 104
    map_pc[19][25] = 105
    map_pc[19][26] = 106

    map_pc[20][8] = 107 # Tv monitor
    
    if animate is None:
        classes = list(range(1, 21))
    elif animate:
        classes = [3, 8, 10, 12, 13, 15, 17]
    else:
        classes = [1, 2, 4, 5, 6, 7, 9, 11, 14, 16, 18, 19, 20]

    return map_pc, classes

def get_relevant_parts(animate):
    if animate:
        return list(range(10, 18)) + list(range(36, 45)) + list(range(46, 54)) + list(range(55, 65)) + list(range(65, 73)) + list(range(77, 89)) + list(range(91, 99))
    else:
        return list(range(1, 6)) + list(range(18, 36)) + [54] + list(range(73, 77)) + [89, 90] + list(range(99, 108))



from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32)
        obj = np.array(obj).astype(np.float32)
        part = np.array(part).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'obj': obj,
                'part': part,
                'fb': fb,
                'edge': edge}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        obj = np.array(obj).astype(np.float32)
        part = np.array(part).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)

        img_obj = img.copy()

        img = np.vstack([img, fb[None, :, :]])
        img = np.vstack([img, edge[None, :, :]])

        img = torch.from_numpy(img).float()
        obj = torch.from_numpy(obj).float()
        part = torch.from_numpy(part).float()

        return {'image': img,
                'obj': obj,
                'part': part,
                'image_obj': img_obj}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            obj = obj.transpose(Image.FLIP_LEFT_RIGHT)
            part = part.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'obj': obj,
                'part': part}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        obj = obj.rotate(rotate_degree, Image.NEAREST)
        part = part.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'obj': obj,
                'part': part}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'obj': obj,
                'part': part}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        obj = obj.resize((ow, oh), Image.NEAREST)
        part = part.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            obj = ImageOps.expand(obj, border=(0, 0, padw, padh), fill=self.fill)
            part = ImageOps.expand(part, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        obj = obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        part = part.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'obj': obj,
                'part': part}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        obj = obj.resize((ow, oh), Image.NEAREST)
        part = part.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        obj = obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        part = part.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'obj': obj,
                'part': part}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']

        # assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        obj = obj.resize(self.size, Image.NEAREST)
        part = part.resize(self.size, Image.NEAREST)

        return {'image': img,
                'obj': obj,
                'part': part}
    
class ResizeMasks(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        obj = sample['obj']
        part = sample['part']
        fb = sample['fb']
        edge = sample['edge']

        w, h = img.size
        short_size = 0
        if w > h:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
            short_size = oh
        else:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
            short_size = ow
            
        img = img.resize((ow, oh), Image.BILINEAR)
        obj = obj.resize((ow, oh), Image.NEAREST)
        part = part.resize((ow, oh), Image.NEAREST)
        fb = fb.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)
        
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            obj = ImageOps.expand(obj, border=(0, 0, padw, padh), fill=0)
            part = ImageOps.expand(part, border=(0, 0, padw, padh), fill=0)
            fb = ImageOps.expand(fb, border=(0, 0, padw, padh), fill=0)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=0)
        
        return {'image': img,
                'obj': obj,
                'part': part,
                'fb': fb,
                'edge': edge}


class SegmentationDataset(Dataset):
    def __init__(self, folder, mode='train'):

        self.folder = folder
        with open(folder + mode + '.txt') as f:
            self.image_path_list = f.read().splitlines()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, i):

        image_path = self.folder + 'JPEGImages/' + self.image_path_list[i] + '.jpg'
        part_label_path = self.folder + 'GT_part_108/' + self.image_path_list[i] + '.png'
        obj_label_path = self.folder + 'object/' + self.image_path_list[i] + '.png'
        fb_label_path = self.folder + 'fb_from_obj_108/' + self.image_path_list[i] + '.png'
        edge_label_path = self.folder + 'hed_edges_108_2/' + self.image_path_list[i] + '.png'

        sample = {}
        sample['image'] = Image.open(image_path)
        org_img = sample['image'].copy()
        org_size = sample['image'].size

        part = np.array(Image.open(part_label_path))
        part = part - (147 * (part == 255))
        sample['part'] = Image.fromarray(part.astype(np.float))
        sample['obj'] = Image.open(obj_label_path)

        sample['fb'] = Image.open(fb_label_path)
        sample['edge'] = Image.open(edge_label_path)

        sample = self.transform_val(sample)

        sample['path'] = self.image_path_list[i]
        sample['orgsize'] = org_size
        sample['org_img'] = np.array(org_img)

        image_name = self.image_path_list[i] + '.png'
        sample['name'] = image_name

        return sample

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            RandomHorizontalFlip(),
            RandomScaleCrop(base_size=513, crop_size=513),
            RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            ResizeMasks(crop_size=770),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

        return composed_transforms(sample)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################


PATH = '/ssd_scratch/cvit/pranav.g/float/'
batch_size = 1

train_dataset = SegmentationDataset(PATH)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
valid_dataset = SegmentationDataset(PATH, mode='val')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)


obj_model = DeepLab(backbone='resnet101', num_classes=21)
model = DeepLabFactored(backbone='resnet101', num_anim_classes=20, num_inanim_classes=27)

obj_model.load_state_dict(torch.load('/home2/pranavgupta77/obj.pth')['state_dict'])
model.load_state_dict(torch.load('/ssd_scratch/cvit/pranav.g/float/ckpt/108_combined_v2_104.pth'))

gpu_ids = [0,1]

if torch.cuda.device_count() > 1:
    obj_model = torch.nn.DataParallel(obj_model, device_ids=gpu_ids)
    patch_replication_callback(obj_model)
    obj_model.cuda()

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    patch_replication_callback(model)
    model.cuda()


def jaccard_perpart_perimg(y_pred, y_true, num_classes):
    y_pred = torch.Tensor(y_pred).type(torch.LongTensor)
    y_true = torch.Tensor(y_true).type(torch.LongTensor)
    y_pred = F.one_hot(y_pred, num_classes=num_classes)
    y_true = F.one_hot(y_true, num_classes=num_classes+1)
    ious = {}
    counts = {}
    for i in range(num_classes):
        pred = y_pred[:,:,i]
        gt = y_true[:,:,i]
        inter = torch.logical_and(pred, gt)
        union = torch.logical_or(pred, gt)
        iou = torch.sum(inter, [0,1]) / torch.sum(union, [0,1])
        legal = torch.sum(gt, [0,1]) > 0
        ious[i] = torch.sum(iou[legal])
        counts[i] = torch.sum(legal)

    return ious, counts


def pad_and_square(x_min, y_min, x_max, y_max, pad, orgsize):
    x_min = max(x_min - pad, 0)
    y_min = max(y_min - pad, 0)
    x_max = min(x_max + pad, orgsize)
    y_max = min(y_max + pad, orgsize)
    
    y_dis = y_max - y_min
    x_dis = x_max - x_min
    
    if y_dis > x_dis:
        diff = y_dis - x_dis
        dsub = diff // 2
        dadd = diff - dsub

        if dsub > x_min:
            x_max = min(x_max + dadd + (dsub - x_min), orgsize)
            x_min = 0
        elif x_max + dadd > orgsize:
            x_min = max(x_min - dsub - (x_max + dadd - orgsize), 0)
            x_max = orgsize
        else:
            x_min = x_min - dsub
            x_max = x_max + dadd

    elif x_dis > y_dis:
        diff = x_dis - y_dis
        dsub = diff // 2
        dadd = diff - dsub
        
        if dsub > y_min:
            y_max = min(y_max + dadd + (dsub - y_min), orgsize)
            y_min = 0
        elif y_max + dadd > orgsize:
            y_min = max(y_min - dsub - (y_max + dadd - orgsize), 0)
            y_max = orgsize
        else:
            y_min = y_min - dsub
            y_max = y_max + dadd

    return x_min, y_min, x_max, y_max


def bbox(img):
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0

    for i in img:
        if np.count_nonzero(i) is not 0:
            break
        y_min+=1

    for i in img.T:
        if np.count_nonzero(i) is not 0:
            break
        x_min+=1

    for i in img[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        y_max+=1
    y_max = img.shape[0] - y_max - 1

    for i in img.T[::-1]:
        if np.count_nonzero(i) is not 0:
            break
        x_max+=1
    x_max = img.shape[1] - x_max - 1

    return x_min, y_min, x_max, y_max


def pred_to_orgsize(part, org_size):
    ow, oh = org_size
    h, w = part.shape
    short_size = 0
    if ow > oh:
        s = int(1.0 * oh * w / ow)
        part = part[:s, :]
    else:
        s = int(1.0 * ow * h / oh)
        part = part[:, :s]

    part = part.astype(np.uint8)
    part = Image.fromarray(part)
    part = part.resize((ow, oh), Image.NEAREST)
    part = np.array(part).astype(np.uint8)

    return part


def combine_obj_part_pred(objs, parts, animate):
    map_pc, classes = part_obj_to_datasetclass(animate=animate)
    preds = np.zeros(objs.shape)
    
    for objkey in classes:
        for partkey in map_pc[objkey]:
            finalkey = map_pc[objkey][partkey]
            obj = (objs == objkey)
            obj = obj.astype(int)
            part = (parts == partkey)
            part = part.astype(int)

            preds += finalkey * (obj * part)

    return preds


def combine_obj_all_parts(objs, anim_parts, inanim_parts):
    anim_final = combine_obj_part_pred(objs, anim_parts, True)
    inanim_final = combine_obj_part_pred(objs, inanim_parts, False)
    
    anim_loc = anim_final > 0
    anim_loc = anim_loc.astype(float)
    inanim_loc = inanim_final > 0
    inanim_loc = inanim_loc.astype(float)
    
    invalid_loc = anim_loc * inanim_loc
    valid_loc = 1 - invalid_loc
    
    anim_final = anim_final * valid_loc
    inanim_final = inanim_final * valid_loc
    
    return anim_final + inanim_final
#################################################################################################################################
#################################################################################################################################
# FLOAT (without IZR)
"""
obj_model.eval()
model.eval()
num_classes = 108
valid_alliou_avg = Evaluator(num_classes)
valid_miou_avg = mIOUMeter(num_classes)

output_dir = '/home2/deepti.rawat/pranav/codes/21/aspp_low_level/outputs/108_v1.1/'
i = 0
tabr = tqdm(valid_dataloader)
for sample in tabr:
    i += 1
    images = sample['image'].float()
    parts = sample['part'].type(torch.LongTensor)
    orgsizes = sample['orgsize']
    names = sample['name']
    nb = images.shape[0]
    images = images.cuda()
    parts = parts.cuda()
    
    objpred = obj_model(images)
    anim_pred, inanim_pred = model(images)

    parts = parts.cpu().detach().numpy()

    anim_pred = anim_pred.cpu().detach().numpy()
    anim_pred = np.argmax(anim_pred, 1)
    
    inanim_pred = inanim_pred.cpu().detach().numpy()
    inanim_pred = np.argmax(inanim_pred, 1)

    objpred = objpred.cpu().detach().numpy()
    objpred = np.argmax(objpred, 1)

    preds = combine_obj_all_parts(objpred, anim_pred, inanim_pred)
    preds = preds.astype(int)

    for j in range(nb):
        pred = pred_to_orgsize(preds[j], (orgsizes[0][j], orgsizes[1][j]))
        gt = pred_to_orgsize(parts[j], (orgsizes[0][j], orgsizes[1][j]))

        ious, counts = jaccard_perpart_perimg(pred, gt, num_classes)
        for cl in range(num_classes):
            ious[cl] = ious[cl].item()
            counts[cl] = counts[cl].item()
        valid_miou_avg.update(ious, counts)

        valid_alliou_avg.add_batch(gt, pred)

        name = names[j]
        pred = Image.fromarray(pred)
        pred.save(output_dir + name)

    

print('Final', valid_miou_avg.avg, valid_alliou_avg.Mean_Intersection_over_Union())"""

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

# FLOAT (with IZR)
pad = 50
inpsize = 770
size = (513, 513)
resize_tr = transforms.Compose([transforms.Resize(size)])
resize_tr_fb_edge = transforms.Compose([transforms.Resize(size,interpolation=Image.NEAREST)])


output_dir = '/home2/pranavgupta77/codes/108/wacv/outputs/108_combined_v2/'
num_classes = 108
obj_model.eval()
model.eval()
valid_alliou_avg = Evaluator(num_classes)
valid_miou_avg = mIOUMeter(num_classes)

i = 0
tabr = tqdm(valid_dataloader)
for sample in tabr:
    i += 1
    images = sample['image'].float()
    parts = sample['part'].type(torch.LongTensor)
    orgsizes = sample['orgsize']
    names = sample['name']
    images_obj  =sample['image_obj']

    num_batches = images.shape[0]
    images = images.cuda()
    parts = parts.cuda()
    
    objpred = obj_model(images_obj)
    animpred, inanimpred = model(images)

    parts = parts.cpu().detach().numpy()

    animpred = animpred.cpu().detach().numpy()
    animpred_lbls = np.argmax(animpred, 1)
    
    inanimpred = inanimpred.cpu().detach().numpy()
    inanimpred_lbls = np.argmax(inanimpred, 1)

    objpred = objpred.cpu().detach().numpy()
    objpred_lbls = np.argmax(objpred, 1)

    # --------------------------------------ZOOM---------------------------------------

    # for every sample in the batch
    for nb in range(num_batches):
        objpred_classes = np.unique(objpred_lbls[nb])
        zoom_info = []
        zoomed_inp = []
        zoomed_inp_obj = []

        # for every unique object classes in the sample
        for obj_cls in objpred_classes:
            if obj_cls == 0:
                continue
            num_labels, labels = cv2.connectedComponents((objpred_lbls[nb] == obj_cls).astype(np.uint8))

            # for every unique component of the object class
            for ncomp in range(1, num_labels):
                # ignore if component is too small
                if np.sum(labels == ncomp) < 25:
                    continue

                x_min, y_min, x_max, y_max = bbox(labels == ncomp)
                x_min, y_min, x_max, y_max = pad_and_square(x_min, y_min, x_max, y_max, pad, inpsize)
                # print(nb, obj_cls, ncomp, x_min, y_min, x_max, y_max)
                if (y_max-y_min) * (x_max-x_min) > 400*400:
                     continue

                if obj_cls in [3, 8, 10, 12, 13, 15, 17]:
                    zoom_info.append((y_min, y_max, x_min, x_max, True, obj_cls))

                elif obj_cls in [1, 2, 4, 5, 6, 7, 9, 11, 14, 16, 18, 19, 20]:
                    zoom_info.append((y_min, y_max, x_min, x_max, False, obj_cls))

                else:
                    assert(False)

                cropimg_img = resize_tr(images[nb, :3, y_min:y_max, x_min:x_max])
                crop_fb_edge =  resize_tr_fb_edge(images[nb, 3:5, y_min:y_max, x_min:x_max])

                cropimg = torch.cat((cropimg_img, crop_fb_edge), 0)

                cropimg_obj = resize_tr(images_obj[nb, :, y_min:y_max, x_min:x_max])
                
                zoomed_inp.append(cropimg)
                zoomed_inp_obj.append(cropimg_obj)

        if len(zoomed_inp) == 0:
            continue
        if len(zoomed_inp_obj) == 0:
            continue

        num_objs = len(zoom_info)
        zoomed_inp = torch.stack(zoomed_inp)
        zoomed_inp_obj = torch.stack(zoomed_inp_obj)

        for iobj in range(num_objs):
            y_min, y_max, x_min, x_max, animate, obj_cls = zoom_info[iobj]
            objpred_zoom = obj_model(zoomed_inp_obj[iobj:iobj+1])
            animpred_zoom, inanimpred_zoom = model(zoomed_inp[iobj:iobj+1])

            resize_zoomed = transforms.Compose([transforms.Resize((y_max-y_min, x_max-x_min), interpolation=Image.NEAREST)])
            obj_zoom = resize_zoomed(objpred_zoom[0])
            obj_zoom = obj_zoom.cpu().detach().numpy()
            obj_zoom_pred = (np.argmax(obj_zoom, 0) == obj_cls).astype(int)

            objpred[nb, :, y_min:y_max, x_min:x_max] = (obj_zoom_pred * obj_zoom) + ((1 - obj_zoom_pred) * objpred[nb, :, y_min:y_max, x_min:x_max])
            if animate:
                anim_zoom = resize_zoomed(animpred_zoom[0])
                anim_zoom = anim_zoom.cpu().detach().numpy()
                animpred[nb, :, y_min:y_max, x_min:x_max] = (obj_zoom_pred * anim_zoom) + ((1 - obj_zoom_pred) * animpred[nb, :, y_min:y_max, x_min:x_max])
            else:
                inanim_zoom = resize_zoomed(inanimpred_zoom[0])
                inanim_zoom = inanim_zoom.cpu().detach().numpy()
                inanimpred[nb, :, y_min:y_max, x_min:x_max] = (obj_zoom_pred * inanim_zoom) + ((1 - obj_zoom_pred) * inanimpred[nb, :, y_min:y_max, x_min:x_max])

    # -------------------------------------COMBINE--------------------------------------
    
    objpred_ = np.argmax(objpred, 1)
    animpred_ = np.argmax(animpred, 1)
    inanimpred_ = np.argmax(inanimpred, 1)

    pred_ = combine_obj_all_parts(objpred_, animpred_, inanimpred_)
    pred_ = pred_.astype(int)

    for j in range(num_batches):
        pred = pred_to_orgsize(pred_[j], (orgsizes[0][j], orgsizes[1][j]))
        gt = pred_to_orgsize(parts[j], (orgsizes[0][j], orgsizes[1][j]))
        valid_alliou_avg.add_batch(gt, pred)
        
        ious, counts = jaccard_perpart_perimg(pred, gt, num_classes)
        for cl in range(num_classes):
            ious[cl] = ious[cl].item()
            counts[cl] = counts[cl].item()
        valid_miou_avg.update(ious, counts)

        name = names[j]
        pred = Image.fromarray(pred)
        pred.save(output_dir + name)

print(valid_alliou_avg.Mean_Intersection_over_Union_PerClass() )
print(valid_alliou_avg.Mean_Intersection_over_Union())
print(valid_miou_avg.avg)
print(valid_miou_avg.sqiou)
				
