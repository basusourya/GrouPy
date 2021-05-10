import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math
from torch.nn.modules.utils import _pair
from groupy.gconv.make_gconv_indices import *


make_indices_functions = {'c4_z2': make_c4_z2_indices,
                          'c4_p4': make_c4_p4_indices,
                          'P4H2_z2': make_P4H2_z2_indices,
                          'P4H2_P4H2Z2': make_P4H2_P4H2Z2_indices,
                          'P4V2_z2': make_P4V2_z2_indices,
                          'P4V2_P4V2Z2': make_P4V2_P4V2Z2_indices,
                          'P4H2V2_z2': make_P4H2V2_z2_indices,
                          'P4H2V2_P4H2V2Z2': make_P4H2V2_P4H2V2Z2_indices,
                          'H2V2_z2': make_H2V2_z2_indices,
                          'H2V2_H2V2Z2': make_H2V2_H2V2Z2_indices,
                          'H2_z2': make_H2_z2_indices,
                          'H2_H2Z2': make_H2_H2Z2_indices,
                          'V2_z2': make_V2_z2_indices,
                          'V2_V2Z2': make_V2_V2Z2_indices,
                          }


def trans_filter(w, inds):
    inds_reshape = inds.reshape((-1, inds.shape[-1])).astype(np.int64)
    w_indexed = w[:, :, inds_reshape[:, 0].tolist(), inds_reshape[:, 1].tolist(), inds_reshape[:, 2].tolist()]
    w_indexed = w_indexed.view(w_indexed.size()[0], w_indexed.size()[1],
                                    inds.shape[0], inds.shape[1], inds.shape[2], inds.shape[3])
    w_transformed = w_indexed.permute(0, 2, 1, 3, 4, 5)
    return w_transformed.contiguous()


class SplitGConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, input_stabilizer_size=1, output_stabilizer_size=4, inds_transformer='c4_z2'):
        super(SplitGConv2D, self).__init__()
        assert inds_transformer in make_indices_functions.keys()
        self.ksize = kernel_size

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_stabilizer_size = input_stabilizer_size
        self.output_stabilizer_size = output_stabilizer_size
        self.inds_transformer = inds_transformer

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels, self.input_stabilizer_size, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.tw = None
        self.reset_parameters()

        self.inds = self.make_transformation_indices()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def make_transformation_indices(self):
        return make_indices_functions[self.inds_transformer](self.ksize)

    def forward(self, input):
        tw = trans_filter(self.weight, self.inds)
        self.tw = tw
        tw_shape = (self.out_channels * self.output_stabilizer_size,
                    self.in_channels * self.input_stabilizer_size,
                    self.ksize, self.ksize)
        tw = tw.view(tw_shape)

        input_shape = input.size()
        input = input.view(input_shape[0], self.in_channels*self.input_stabilizer_size, input_shape[-2], input_shape[-1])

        y = F.conv2d(input, weight=tw, bias=None, stride=self.stride,
                        padding=self.padding)
        batch_size, _, ny_out, nx_out = y.size()
        y = y.view(batch_size, self.out_channels, self.output_stabilizer_size, ny_out, nx_out)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1, 1)
            y = y + bias

        return y


class P4ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, inds_transformer='c4_z2', *args, **kwargs)

class P4ConvP4(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4ConvP4, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, inds_transformer='c4_p4', *args, **kwargs)

class P4H2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4H2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, inds_transformer='P4H2_z2', *args, **kwargs)

class P4H2ConvP4H2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4H2ConvP4H2, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, inds_transformer='P4H2_P4H2Z2', *args, **kwargs)

class P4V2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4V2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=8, inds_transformer='P4V2_z2', *args, **kwargs)

class P4V2ConvP4V2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4V2ConvP4V2, self).__init__(input_stabilizer_size=8, output_stabilizer_size=8, inds_transformer='P4V2_P4V2Z2', *args, **kwargs)

class P4H2V2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4H2V2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=16, inds_transformer='P4H2V2_z2', *args, **kwargs)

class P4H2V2ConvP4H2V2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(P4H2V2ConvP4H2V2, self).__init__(input_stabilizer_size=16, output_stabilizer_size=16, inds_transformer='P4H2V2_P4H2V2Z2', *args, **kwargs)

class H2V2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(H2V2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=4, inds_transformer='H2V2_z2', *args, **kwargs)

class H2V2ConvH2V2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(H2V2ConvH2V2, self).__init__(input_stabilizer_size=4, output_stabilizer_size=4, inds_transformer='H2V2_H2V2Z2', *args, **kwargs)

class H2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(H2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=2, inds_transformer='H2_z2', *args, **kwargs)

class H2ConvH2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(H2ConvH2, self).__init__(input_stabilizer_size=2, output_stabilizer_size=2, inds_transformer='H2_H2Z2', *args, **kwargs)

class V2ConvZ2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(V2ConvZ2, self).__init__(input_stabilizer_size=1, output_stabilizer_size=2, inds_transformer='V2_z2', *args, **kwargs)

class V2ConvV2(SplitGConv2D):

    def __init__(self, *args, **kwargs):
        super(V2ConvV2, self).__init__(input_stabilizer_size=2, output_stabilizer_size=2, inds_transformer='V2_V2Z2', *args, **kwargs)
