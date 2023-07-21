#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#lib
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
import torch
import math

#pt image tensor to np 
def pt_to_np(tensor):
    return np.ascontiguousarray(tensor.permute(1,2,0).numpy())

#eca (attention layer)
class EcaModule(nn.Module):
    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)

#conv
def Conv2d(*args, **kwargs):
    args = [int(a) if type(a) != tuple else a for i,a in enumerate(args) if i < 6]
    if not 'padding' in kwargs:
        k = args[2] if len(args) > 2 else (kwargs['kernel_size'] if 'kernel_size' in kwargs else kwargs['k'])
        k = (k,k) if type(k) != tuple else k
        pad = ((k[0] - 1) // 2,(k[1] - 1) // 2)
        kwargs['padding'] = pad
    return nn.Conv2d(*args, **kwargs, **{'padding_mode':'zeros'})

#conv > bn > act
class convolution(nn.Module):
    def __init__(self, inp_dim, out_dim, k=3, stride=1, groups=1, bn=True, act=True, dilation=1, bias=True, **kwargs):
        super(convolution, self).__init__()
        self.conv = Conv2d(inp_dim, out_dim, k, stride=(stride, stride), bias=not bn and bias, groups=groups, dilation=dilation, **kwargs)
        self.bn   = nn.BatchNorm2d(out_dim) if bn else nn.Identity()
        self.activation = nn.ReLU(True) if act else nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        return out

#conv > bn > act > att
class convolution_att(convolution):
    def __init__(self, inp_dim, out_dim, k=3, stride=1, groups=1, bn=True, act=True, dilation=1, attention='eca'):
        super(convolution_att, self).__init__(inp_dim, out_dim, k, stride, groups, bn, act, dilation)
        self.attention = EcaModule(out_dim)
    def forward(self, x):
        out = super().forward(x)
        out = self.attention(out)
        return out

#model (ultrasimplified u-net like)
class JWSTFixModel(nn.Module):
    def __init__(self, f=8, lay = convolution_att):
        super().__init__()
        self.normalize_t = T.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757)) 
        upsample_f = nn.Upsample(scale_factor=2)
        self.down_conv1 = nn.Sequential(*[lay(3, f*2, stride=2)])
        self.down_conv2 = nn.Sequential(*[lay(f*2, f*4, stride=2)])
        self.up_conv2 = nn.Sequential(*[lay(f*4, f*2), upsample_f])
        self.up_conv1 = nn.Sequential(*[lay(f*2, f*2), upsample_f])
        self.end_conv = nn.Sequential(*[lay(f*2,f,1), convolution(f,3,3,act=False)])
    
    def forward(self, x):
        in_x = x
        x = self.normalize_t(x)
        x = self.down_conv1(x)
        x = self.down_conv2(x)
        x = self.up_conv2(x)
        x = self.up_conv1(x)
        x = self.end_conv(x)
        x = 1-(x*20).sigmoid() 
        x = T.functional.resize(x, in_x.shape[2:])
        x = torch.clamp_(x, 0, 1) * in_x
        return x




