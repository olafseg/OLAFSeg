
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
import matplotlib.pyplot as plt
#import seaborn as sns

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torchinfo import summary
import torch.utils.model_zoo as model_zoo
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

"""Synchronised BN"""

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

"""ResNet"""

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


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs



class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm,
                 pretrained=True):
        super(Xception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        #x = self.conv4(x)
        #x = self.bn4(x)
        #x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

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
        pretrain_dict = model_zoo.load_url('https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_xception_71-8eec7df1.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)










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

        pretrained_weights = pretrain_dict['conv1.weight']
        modified_weights = torch.zeros_like(state_dict['conv1.weight'])
        modified_weights[:, :3, :, :] = pretrained_weights
        modified_weights[:, 3, :, :] = torch.mean(pretrained_weights[:, :3, :, :], dim = 1)
        modified_weights[:, 4, :, :] = torch.mean(pretrained_weights[:, :3, :, :], dim = 1)
        state_dict['conv1.weight'] = modified_weights

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k == 'conv1.weight':# or k == 'bn1.running_mean' or k == 'bn1.running_var' or k == 'bn1.weight' or k == 'bn1.bias':
                    continue
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict, strict = False)

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

"""Backbone"""

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return ResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return Xception(output_stride, BatchNorm, pretrained=True)
    else:
        raise NotImplementedError

"""ASPP"""

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
        #print(x1.size(), x2.size(), x3.size(), x4.size(), x5.size())
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

"""Decoder"""

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
    
###low level feature extractor ####################

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


"""DeepLab"""

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

#model = DeepLabFactored(backbone='xception', num_anim_classes=24, num_inanim_classes=29)


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

def part_obj_to_datasetclass(obj_classes=21, animate=True):

    map_pc = {}
    for i in range(obj_classes):
        map_pc[i] = {}
        
    # Animate objects

    map_pc[3][1] = 8  # Bird
    map_pc[3][5] = 9
    map_pc[3][3] = 10
    map_pc[3][2] = 11

    map_pc[8][1] = 23  # Cat
    map_pc[8][3] = 24
    map_pc[8][4] = 25
    map_pc[8][2] = 26

    map_pc[10][1] = 28  # Cow
    map_pc[10][4] = 29
    map_pc[10][3] = 30
    map_pc[10][2] = 31

    map_pc[12][1] = 33  # Dog
    map_pc[12][3] = 34
    map_pc[12][4] = 35
    map_pc[12][2] = 36

    map_pc[13][1] = 37  # Horse
    map_pc[13][4] = 38
    map_pc[13][3] = 39
    map_pc[13][2] = 40

    map_pc[15][1] = 43  # Person
    map_pc[15][2] = 44
    map_pc[15][7] = 45
    map_pc[15][6] = 46
    map_pc[15][8] = 47
    map_pc[15][3] = 48

    map_pc[17][1] = 51  # Sheep
    map_pc[17][3] = 52
    map_pc[17][2] = 53

    
    # Inanimate objects
    map_pc[1][1] = 1  # Aeroplane
    map_pc[1][5] = 2
    map_pc[1][3] = 3
    map_pc[1][4] = 4
    map_pc[1][2] = 5
    
    map_pc[2][2] = 6 # Bicycle
    map_pc[2][1] = 7
    
    map_pc[4][0] = 12  # Boat

    map_pc[5][13] = 13 # Bottle
    map_pc[5][14] = 14

    map_pc[6][12] = 15 # Bus
    map_pc[6][2] = 16
    map_pc[6][1] = 17

    map_pc[7][12] = 18 # Car
    map_pc[7][2] = 19
    map_pc[7][6] = 20
    map_pc[7][7] = 21
    map_pc[7][1] = 22
    
    map_pc[9][0] = 27  # Chair
    
    map_pc[11][0] = 32  # Dining Table

    map_pc[14][2] = 41  # Motorbike
    map_pc[14][1] = 42

    map_pc[16][10] = 49  # Potted plant
    map_pc[16][11] = 50
    
    map_pc[18][0] = 54  # Sofa

    map_pc[19][0] = 55  # Train

    map_pc[20][8] = 56  # Tv monitor
    map_pc[20][9] = 57
    
    if animate is None:
        classes = list(range(1, 21))
    elif animate:
        classes = [3, 8, 10, 12, 13, 15, 17]
    else:
        classes = [1, 2, 4, 5, 6, 7, 9, 11, 14, 16, 18, 19, 20]

    return map_pc, classes

def get_relevant_parts(animate):
    if animate:
        return [8, 9, 10, 11, 23, 24, 25, 26, 28, 29, 30, 31, 33, 34,
                35, 36, 37, 38, 39, 40, 43, 44, 45, 46, 47, 48, 51, 52, 53]
    else:
        return [1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19,
                20, 21, 22, 27, 32, 41, 42, 49, 50, 54, 55, 56, 57]

def aggregate_parts_to_classes(num_classes=58, level=1, animate=True):
    
    # Level 1 animate parts : Head (1), Torso (2), (Upper) Leg (3), Tail (4), Wing(5),
    #                         Upper Arm (6), Lower Arm(7), Lower Leg(8)

    map_pc = {}
    for i in range(num_classes):
        map_pc[i] = 0

    if animate:
        map_pc[8] = 1  # Bird
        map_pc[9] = 5
        map_pc[10] = 3
        map_pc[11] = 2

        map_pc[23] = 1  # Cat
        map_pc[24] = 3
        map_pc[25] = 4
        map_pc[26] = 2

        map_pc[28] = 1  # Cow
        map_pc[29] = 4
        map_pc[30] = 3
        map_pc[31] = 2

        map_pc[33] = 1  # Dog
        map_pc[34] = 3
        map_pc[35] = 4
        map_pc[36] = 2

        map_pc[37] = 1  # Horse
        map_pc[38] = 4
        map_pc[39] = 3
        map_pc[40] = 2

        map_pc[43] = 1  # Person
        map_pc[44] = 2
        map_pc[45] = 7
        map_pc[46] = 6
        map_pc[47] = 8
        map_pc[48] = 3

        map_pc[51] = 1  # Sheep
        map_pc[52] = 3
        map_pc[53] = 2
        
    # Level 1 inanimate parts : Body (1), Wheel (2), Wing (3), Stern (4), Engine(5), Light (6)
    #                           Plate (7), Screen (8), Frame (9), Pot (10), Plant (11), Window (12),
    #                           Bottle Cap (13), Bottle Body (14)

    else:
        map_pc[1] = 1  # Aeroplane
        map_pc[2] = 5
        map_pc[3] = 3
        map_pc[4] = 4
        map_pc[5] = 2

        map_pc[6] = 2 # Bicycle
        map_pc[7] = 1

        map_pc[13] = 13 # Bottle
        map_pc[14] = 14

        map_pc[15] = 12 # Bus
        map_pc[16] = 2
        map_pc[17] = 1

        map_pc[18] = 12  # Car
        map_pc[19] = 2
        map_pc[20] = 6
        map_pc[21] = 7
        map_pc[22] = 1

        map_pc[41] = 2 # Motorbike
        map_pc[42] = 1

        map_pc[49] = 10 # Potted plant
        map_pc[50] = 11

        map_pc[56] = 8 # Tv monitor
        map_pc[57] = 9

    return map_pc

def parts_to_object_class(num_classes=58, level=1):

    map_pc = {}
    for i in range(num_classes):
        map_pc[i] = 0

    if animate:
        map_pc[8] = 3  # Bird
        map_pc[9] = 3
        map_pc[10] = 3
        map_pc[11] = 3

        map_pc[23] = 8  # Cat
        map_pc[24] = 8
        map_pc[25] = 8
        map_pc[26] = 8

        map_pc[28] = 10  # Cow
        map_pc[29] = 10
        map_pc[30] = 10
        map_pc[31] = 10

        map_pc[33] = 12  # Dog
        map_pc[34] = 12
        map_pc[35] = 12
        map_pc[36] = 12

        map_pc[37] = 13  # Horse
        map_pc[38] = 13
        map_pc[39] = 13
        map_pc[40] = 13

        map_pc[43] = 15  # Person
        map_pc[44] = 15
        map_pc[45] = 15
        map_pc[46] = 15
        map_pc[47] = 15
        map_pc[48] = 15

        map_pc[51] = 17  # Sheep
        map_pc[52] = 17
        map_pc[53] = 17

        map_pc[1] = 1  # Aeroplane
        map_pc[2] = 1
        map_pc[3] = 1
        map_pc[4] = 1
        map_pc[5] = 1

        map_pc[6] = 2 # Bicycle
        map_pc[7] = 2

        map_pc[13] = 5 # Bottle
        map_pc[14] = 5

        map_pc[15] = 6 # Bus
        map_pc[16] = 6
        map_pc[17] = 6

        map_pc[18] = 7  # Car
        map_pc[19] = 7
        map_pc[20] = 7
        map_pc[21] = 7
        map_pc[22] = 7

        map_pc[41] = 14 # Motorbike
        map_pc[42] = 14

        map_pc[49] = 16 # Potted plant
        map_pc[50] = 16

        map_pc[56] = 20 # Tv monitor
        map_pc[57] = 20

    return map_pc

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
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32)
        anim_obj = np.array(anim_obj).astype(np.float32)
        inanim_obj = np.array(inanim_obj).astype(np.float32)
        anim = np.array(anim).astype(np.float32)
        inanim = np.array(inanim).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        anim_obj = np.array(anim_obj).astype(np.float32)
        inanim_obj = np.array(inanim_obj).astype(np.float32)
        anim = np.array(anim).astype(np.float32)
        inanim = np.array(inanim).astype(np.float32)
        fb = np.array(fb).astype(np.float32)
        edge = np.array(edge).astype(np.float32)
        


        img = np.vstack([img, fb[None, :, :]])
        img = np.vstack([img, edge[None, :, :]])

        #print(np.shape(img))

        img = torch.from_numpy(img).float()
        anim_obj = torch.from_numpy(anim_obj).float()
        inanim_obj = torch.from_numpy(inanim_obj).float()
        anim = torch.from_numpy(anim).float()
        inanim = torch.from_numpy(inanim).float()

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            anim_obj = anim_obj.transpose(Image.FLIP_LEFT_RIGHT)
            inanim_obj = inanim_obj.transpose(Image.FLIP_LEFT_RIGHT)
            anim = anim.transpose(Image.FLIP_LEFT_RIGHT)
            inanim = inanim.transpose(Image.FLIP_LEFT_RIGHT)
            fb = fb.transpose(Image.FLIP_LEFT_RIGHT)
            edge = edge.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        anim_obj = anim_obj.rotate(rotate_degree, Image.NEAREST)
        inanim_obj = inanim_obj.rotate(rotate_degree, Image.NEAREST)
        anim = anim.rotate(rotate_degree, Image.NEAREST)
        inanim = inanim.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']
        fb = sample['fb']
        edge = sample['edge']

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
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)
        fb = fb.resize((ow, oh), Image.NEAREST)
        edge = edge.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            anim_obj = ImageOps.expand(anim_obj, border=(0, 0, padw, padh), fill=self.fill)
            inanim_obj = ImageOps.expand(inanim_obj, border=(0, 0, padw, padh), fill=self.fill)
            anim = ImageOps.expand(anim, border=(0, 0, padw, padh), fill=self.fill)
            inanim = ImageOps.expand(inanim, border=(0, 0, padw, padh), fill=self.fill)
            fb = ImageOps.expand(fb, border=(0, 0, padw, padh), fill=self.fill)
            edge = ImageOps.expand(edge, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim_obj = anim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim_obj = inanim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim = anim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim = inanim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        fb = fb.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        edge = edge.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))



        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim,
                'fb': fb,
                'edge': edge}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim_obj = anim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim_obj = inanim_obj.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        anim = anim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        inanim = inanim.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

        # assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        anim_obj = anim_obj.resize(self.size, Image.NEAREST)
        inanim_obj = inanim_obj.resize(self.size, Image.NEAREST)
        anim = anim.resize(self.size, Image.NEAREST)
        inanim = inanim.resize(self.size, Image.NEAREST)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}
    
class ResizeMasks(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        anim_obj = sample['anim_obj']
        inanim_obj = sample['inanim_obj']
        anim = sample['anim']
        inanim = sample['inanim']

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
        anim_obj = anim_obj.resize((ow, oh), Image.NEAREST)
        inanim_obj = inanim_obj.resize((ow, oh), Image.NEAREST)
        anim = anim.resize((ow, oh), Image.NEAREST)
        inanim = inanim.resize((ow, oh), Image.NEAREST)

        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            anim_obj = ImageOps.expand(anim_obj, border=(0, 0, padw, padh), fill=0)
            inanim_obj = ImageOps.expand(inanim_obj, border=(0, 0, padw, padh), fill=0)
            anim = ImageOps.expand(anim, border=(0, 0, padw, padh), fill=0)
            inanim = ImageOps.expand(inanim, border=(0, 0, padw, padh), fill=0)

        return {'image': img,
                'anim_obj': anim_obj,
                'inanim_obj': inanim_obj,
                'anim': anim,
                'inanim': inanim}

class SegmentationDataset(Dataset):
    def __init__(self, folder, mode='train',
                 input_shape=(513, 513), num_classes=58):

        with open(folder + mode + '.txt') as f:
            self.image_path_list = f.read().splitlines()

        self.input_shape = input_shape
        self.mode = mode
        self.folder = folder
        self.num_classes = num_classes
        self.anim_aggregation_map = aggregate_parts_to_classes(num_classes, animate=True)
        self.inanim_aggregation_map = aggregate_parts_to_classes(num_classes, animate=False)
        self.anim_classes = [3, 8, 10, 12, 13, 15, 17]
        self.inanim_classes = [1, 2, 5, 6, 7, 14, 16, 20]

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, i):
        
        image_path = self.folder + 'JPEGImages/' + self.image_path_list[i] + '.jpg'
        part_label_path = self.folder + 'GT_part_58/' + self.image_path_list[i] + '.png'
        obj_label_path = self.folder + 'object/' + self.image_path_list[i] + '.png'
        fb_label_path = self.folder + 'fb_from_obj_58/' + self.image_path_list[i] + '.png'
        edge_label_path = self.folder + 'hed_edges_58_2/' + self.image_path_list[i] + '.png'

        sample = {}
        sample['image'] = Image.open(image_path)
        org_size = sample['image'].size

        part_label = np.array(Image.open(part_label_path))
        anim_label = self.aggregate_anim_labels(part_label)
        sample['anim'] = Image.fromarray(anim_label)
        inanim_label = self.aggregate_inanim_labels(part_label)
        sample['inanim'] = Image.fromarray(inanim_label)

        obj_label = np.array(Image.open(obj_label_path))
        anim_obj_label = self.anim_remove_objs(obj_label)
        sample['anim_obj'] = Image.fromarray(anim_obj_label)
        inanim_obj_label = self.inanim_remove_objs(obj_label)
        sample['inanim_obj'] = Image.fromarray(inanim_obj_label)

        sample['fb'] = Image.open(fb_label_path)
        sample['edge'] = Image.open(edge_label_path)

        sample = self.transform_tr(sample)

        sample['path'] = self.image_path_list[i]
        sample['orgsize'] = org_size

        return sample

    def anim_remove_objs(self, obj_label):
        final_label = np.zeros(obj_label.shape)
        for i in self.anim_classes:
            obj = (obj_label == i)
            obj = obj.astype(float)
            final_label += (obj * i)

        return final_label
    
    def inanim_remove_objs(self, obj_label):
        final_label = np.zeros(obj_label.shape)
        for i in self.inanim_classes:
            obj = (obj_label == i)
            obj = obj.astype(float)
            final_label += (obj * i)

        return final_label

    def aggregate_anim_labels(self, part_label):
        final_label = np.zeros(part_label.shape)
        for i in range(self.num_classes):
            part = (part_label == i)
            part = part.astype(float)
            final_label += self.anim_aggregation_map[i] * part

        return final_label
    
    def aggregate_inanim_labels(self, part_label):
        final_label = np.zeros(part_label.shape)
        for i in range(self.num_classes):
            part = (part_label == i)
            part = part.astype(float)
            final_label += self.inanim_aggregation_map[i] * part

        return final_label

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



num_anim_classes = 9
num_inanim_classes = 15

PATH = '/ssd_scratch/cvit/pranav.g/float/'
batch_size = 11

train_dataset = SegmentationDataset(PATH)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
valid_dataset = SegmentationDataset(PATH, mode='val')
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

model = DeepLabFactored(backbone='resnet101', num_anim_classes=num_anim_classes, num_inanim_classes=num_inanim_classes)

"""from torchsummary import summary
model.cuda()
print(summary(model, ( 4,513,513)))"""


gpu_ids = [0,1,2,3]
lr = 0.005
num_epochs = 105

train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                {'params': model.get_10x_lr_params(), 'lr': lr * 10}]

optimizer = optim.SGD(train_params, momentum=0.9, weight_decay=1e-4)
lr_scheduler = LR_Scheduler('poly', lr, num_epochs, len(train_dataloader), warmup_epochs=5)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    patch_replication_callback(model)
    model.cuda()

model.train()

for epoch in range(num_epochs):
    epoch_start = time.time()
    train_loss_avg = AverageMeter()
    anim_miou_avg = mIOUMeter(num_anim_classes)
    anim_alliou_avg = Evaluator(num_anim_classes)
    inanim_miou_avg = mIOUMeter(num_inanim_classes)
    inanim_alliou_avg = Evaluator(num_inanim_classes)

    tbar = tqdm(train_dataloader)
    for i, sample in enumerate(tbar):
        images = sample['image'].float()

        anim_objs = sample['anim_obj'].type(torch.LongTensor)
        anim_objs = anim_objs > 0
        anim_objs = anim_objs.type(torch.LongTensor)

        inanim_objs = sample['inanim_obj'].type(torch.LongTensor)
        inanim_objs = inanim_objs > 0
        inanim_objs = inanim_objs.type(torch.LongTensor)

        anims = sample['anim'].type(torch.LongTensor)
        inanims = sample['inanim'].type(torch.LongTensor)

        nb = images.shape[0]

        images = images.cuda()
        anim_objs = anim_objs.cuda()
        inanim_objs = inanim_objs.cuda()
        anims = anims.cuda()
        inanims = inanims.cuda()

        lr_scheduler(optimizer, i, epoch)
        optimizer.zero_grad()
        anim_pred, inanim_pred = model(images)

        anim_ce_loss = crossentropy_loss(anim_pred, anims)
        anim_graph_loss = adjacent_graph_loss(anim_pred, anims, num_anim_classes)
        anim_obj_loss = objmask_loss(anim_pred, anim_objs, num_anim_classes)
        anim_loss = anim_ce_loss + anim_graph_loss + anim_obj_loss
        
        inanim_ce_loss = crossentropy_loss(inanim_pred, inanims)
        inanim_graph_loss = adjacent_graph_loss(inanim_pred, inanims, num_inanim_classes)
        inanim_obj_loss = objmask_loss(inanim_pred, inanim_objs, num_inanim_classes)
        inanim_loss = inanim_ce_loss + inanim_graph_loss + inanim_obj_loss

        loss = anim_loss + inanim_loss

        ious, counts = jaccard_perpart(anim_pred, anims, num_anim_classes)
        for cl in range(num_anim_classes):
            ious[cl] = ious[cl].item()
            counts[cl] = counts[cl].item()
        anim_miou_avg.update(ious, counts)

        ious, counts = jaccard_perpart(inanim_pred, inanims, num_inanim_classes)
        for cl in range(num_inanim_classes):
            ious[cl] = ious[cl].item()
            counts[cl] = counts[cl].item()
        inanim_miou_avg.update(ious, counts)

        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        train_loss_avg.update(train_loss, nb)

        anims_ = anims.cpu().detach().numpy()
        anim_pred_ = anim_pred.cpu().detach().numpy()
        anim_pred_ = np.argmax(anim_pred_, 1)
        anim_alliou_avg.add_batch(anims_, anim_pred_)

        inanims_ = inanims.cpu().detach().numpy()
        inanim_pred_ = inanim_pred.cpu().detach().numpy()
        inanim_pred_ = np.argmax(inanim_pred_, 1)
        inanim_alliou_avg.add_batch(inanims_, inanim_pred_)

        tbar.set_description('loss:%.3f, loss avg::%.3f, pq.a:%.3f, m.a:%.3f, pq.i:%.3f, m.i:%.3f' % (loss.item(), train_loss_avg.avg, anim_miou_avg.avg,
                                                                   anim_alliou_avg.Mean_Intersection_over_Union(),
                                                                   inanim_miou_avg.avg,
                                                                   inanim_alliou_avg.Mean_Intersection_over_Union()))

    epoch_end = time.time()

    print('Epoch:', epoch)
    print('Loss    : {:.5f}'.format(loss.item()))
    print('Loss Avg: {:.5f}'.format(train_loss_avg.avg))

    print('Anim pqIOU   : {:.5f}'.format(anim_miou_avg.avg))
    print('Anim mIOU    : {:.5f}'.format(anim_alliou_avg.Mean_Intersection_over_Union()))

    print('Inanim pqIOU   : {:.5f}'.format(inanim_miou_avg.avg))
    print('Inanim mIOU    : {:.5f}'.format(inanim_alliou_avg.Mean_Intersection_over_Union()))

    print("Epoch time taken:", str(epoch_end-epoch_start))
    print('----------------------------------------')
    
    torch.save(model.module.state_dict(), '/ssd_scratch/cvit/pranav.g/float/ckpt/58_combined_v2_' + str(epoch) + '.pth')
