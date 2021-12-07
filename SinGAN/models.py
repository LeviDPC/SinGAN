import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from SinGAN.AudioSample import AudioSample


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, batch_norm, dilation, dropout=0,
                 use_RELU=False):
        super(ConvBlock, self).__init__()
        self.add_module('conv',
                        nn.Conv1d(in_channel, out_channel, dilation=dilation, kernel_size=ker_size, stride=stride,
                                  padding=padd)),
        # TODO: Get Batch norm working as an optional parameter
        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(out_channel)),
        if use_RELU:
            self.add_module('Relu', nn.LeakyReLU(inplace=True))
        else:
            self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            self.add_module('Dropout', nn.Dropout(p=dropout))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self, opt):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_aud, N, opt.ker_size, opt.padd_size, opt.stride, opt.batch_norm,
                              dilation=opt.dilation)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            if opt.change_channel_count > 0:
                N = int(opt.nfc / pow(2, (i + 1)))
            else:
                N = 1
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, opt.stride,
                              opt.batch_norm, dilation=opt.dilation, dropout=opt.dropout)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv1d(max(N, opt.min_nfc), 1, kernel_size=opt.ker_size, stride=1, dilation=opt.dilation,
                              padding=opt.padd_size)

    def forward(self, x):
        x = self.head(x)  # Layer 1
        x = self.body(x)  # Layers 2-4
        x = self.tail(x)  # Layer5
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        self.opt = opt
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_aud, N, opt.ker_size, opt.padd_size, opt.stride,
                              opt.batch_norm, dilation=opt.dilation,
                              use_RELU=opt.RELU_in_gen)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer - 2):
            if opt.change_channel_count > 0:
                N = int(opt.nfc / pow(2, (i + 1)))
            else:
                N = 1
            block = ConvBlock(max(2 * N, opt.min_nfc), max(N, opt.min_nfc), opt.ker_size, opt.padd_size, opt.stride,
                              opt.batch_norm, dilation=opt.dilation, use_RELU=opt.RELU_in_gen)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv1d(max(N, opt.min_nfc), opt.nc_aud, kernel_size=opt.ker_size, stride=opt.stride,
                      dilation=opt.dilation, padding=opt.padd_size),
            nn.Tanh()
        )

    def forward(self, x, y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # These next two lines exist because y may (and seems to often be, at least for images) bigger
        # Then x. This trims off some off from each edge to make them the same size. Assumes it's the same
        # amount bigger in each dimension

        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind:(y.shape[2] - ind)]
        return x + y
