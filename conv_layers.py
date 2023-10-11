import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import copy

import torchvision.ops
import math

class DeformableElementwiseDynamicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 dynamic = 2,
                 trained_layer = None):

        super(DeformableElementwiseDynamicConv2d, self).__init__()

        self.padding = padding
        self.stride = stride
        self.dynamic = dynamic
        self.offset_size = 2 * kernel_size*kernel_size
        self.entire_size = self.offset_size
        self.out_channels = out_channels

        
        self.offset_modulator_conv1 = nn.Conv2d(in_channels, 
                                     mid_channels,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=bias)
        self.offset_modulator_conv2 = nn.Conv2d(mid_channels, 
                                     mid_channels,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=bias)
        self.offset_modulator_conv3 = nn.Conv2d(mid_channels, 
                                     self.dynamic * self.entire_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=bias)

        nn.init.constant_(self.offset_modulator_conv3.weight, 0.)
        if bias:
            nn.init.constant_(self.offset_modulator_conv3.bias, 0.)


        self.dynamic_attention = nn.Conv2d(in_channels=in_channels,
                                      out_channels=dynamic*out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

        nn.init.constant_(self.dynamic_attention.weight, 0.)
        if bias:
            nn.init.constant_(self.dynamic_attention.bias, 0.)
        
        self.dynamic_conv_list = torch.nn.ModuleList()

        for dk in range(self.dynamic):
            if trained_layer is None:
                self.dynamic_conv_list.append( nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=bias) )
            else:
                self.dynamic_conv_list.append( copy.deepcopy(trained_layer) )



    def forward(self, x):
        offset_modulator = self.offset_modulator_conv1(x)
        offset_modulator = F.relu(offset_modulator)
        offset_modulator = self.offset_modulator_conv2(offset_modulator)
        offset_modulator = F.relu(offset_modulator)
        offset_modulator = self.offset_modulator_conv3(offset_modulator)
        attention_map = self.dynamic_attention( x )
        for dk in range(self.dynamic):
            if dk == 0:
                entire_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
            else:
                current_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
                entire_attention_map = torch.cat( (entire_attention_map, current_attention_map), dim = 0 )
        entire_attention_map = torch.softmax(entire_attention_map, dim = 0)
            
        for dk in range(self.dynamic):
            offset = offset_modulator[:, dk * self.entire_size : dk * self.entire_size + self.offset_size, :, :]
            
            dk_weight = self.dynamic_conv_list[dk].weight
            dk_bias = self.dynamic_conv_list[dk].bias

            dk_feature = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=dk_weight, 
                                          bias=dk_bias, 
                                          padding=self.padding,
                                          stride=self.stride
                                          )
            
            attention_map = entire_attention_map[dk, :, :, :, :]
            if dk == 0:
                dynamic_feature = dk_feature * attention_map
            else:
                dynamic_feature += dk_feature * attention_map

        return dynamic_feature

    def get_attention(self, x):
        attention_map = self.dynamic_attention( x )
        for dk in range(self.dynamic):
            if dk == 0:
                entire_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
            else:
                current_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
                entire_attention_map = torch.cat( (entire_attention_map, current_attention_map), dim = 0 )
        entire_attention_map = torch.softmax(entire_attention_map, dim = 0)
            
        return entire_attention_map


class ElementwiseDynamicConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 dynamic = 2,
                 trained_layer = None):

        super(ElementwiseDynamicConv2d, self).__init__()

        self.padding = padding
        self.stride = stride
        self.dynamic = dynamic
        self.offset_size = 2 * kernel_size*kernel_size
        self.entire_size = self.offset_size
        self.out_channels = out_channels

        self.dynamic_attention = nn.Conv2d(in_channels=in_channels,
                                      out_channels=dynamic*out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

        nn.init.constant_(self.dynamic_attention.weight, 0.)
        if bias:
            nn.init.constant_(self.dynamic_attention.bias, 0.)
        
        self.dynamic_conv_list = torch.nn.ModuleList()

        for dk in range(self.dynamic):
            if trained_layer is None:
                self.dynamic_conv_list.append( nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=bias) )
            else:
                self.dynamic_conv_list.append( copy.deepcopy(trained_layer) )



    def forward(self, x):

        attention_map = self.dynamic_attention( x )
        for dk in range(self.dynamic):
            if dk == 0:
                entire_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
            else:
                current_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
                entire_attention_map = torch.cat( (entire_attention_map, current_attention_map), dim = 0 )
        entire_attention_map = torch.softmax(entire_attention_map, dim = 0)
            

        for dk in range(self.dynamic):


            dk_weight = self.dynamic_conv_list[dk].weight
            dk_bias = self.dynamic_conv_list[dk].bias

            dk_conv = self.dynamic_conv_list[dk]
            dk_feature = dk_conv(x)
            attention_map = entire_attention_map[dk, :, :, :, :]

            if dk == 0:
                dynamic_feature = dk_feature * attention_map
            else:
                dynamic_feature += dk_feature * attention_map

        return dynamic_feature

    def get_attention(self, x):
        attention_map = self.dynamic_attention( x )
        for dk in range(self.dynamic):
            if dk == 0:
                entire_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
            else:
                current_attention_map = attention_map[:, dk*self.out_channels:(dk+1)*self.out_channels, :, :].unsqueeze_(0)
                entire_attention_map = torch.cat( (entire_attention_map, current_attention_map), dim = 0 )
        entire_attention_map = torch.softmax(entire_attention_map, dim = 0)
            
        return entire_attention_map

