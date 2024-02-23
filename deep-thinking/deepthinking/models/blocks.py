
from torch import nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """Basic residual block class 1D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock1DBN(nn.Module):
    """Basic residual block class 1D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.BatchNorm1d(planes) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.BatchNorm1d(planes) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2D(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2D1x1(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=stride, padding=0, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2D5x5(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=5,
                               stride=stride, padding=2, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=5,
                               stride=1, padding=2, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2D7x7(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=7,
                               stride=stride, padding=3, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=7,
                               stride=1, padding=3, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlock2DNoProp(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=stride, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2DNoFinalRelu(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x) ## update could also be here
        # out = F.relu(out)
        return out

class BasicBlock2DPropBegin(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2DPropEnd(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=stride, padding=0, bias=bias)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=bias))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock2DSimple(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(state)
        out = F.relu(out)
        return out


class BasicBlock2DSimple2(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=stride, padding=0, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(state)
        out = F.relu(out)
        return out


class Block1Conv2d(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
        #                        stride=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))



    def forward(self, x, state):
        out = self.conv1(x)
        out += self.shortcut(state)
        out = F.relu(out)
        return out

class Block1Conv2dAdd(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
        #                        stride=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = F.relu(self.conv1(x))
        return out + self.shortcut(state)

        # return out


class BasicBlock2DSimpleTanh(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(state)
        out = F.tanh(out)
        return out



class Block1Conv2dTanh(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
        #                        stride=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = self.conv1(x)
        out += self.shortcut(state)
        out = F.tanh(out)
        return out

class Block1Conv2dAddTanh(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False,bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
        #                        stridnn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')e=1, padding=0, bias=bias)
        # self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        # if stride != 1 or in_planes != self.expansion * planes:
        #     self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * planes,
        #                                             kernel_size=1, stride=stride, bias=False))

    def forward(self, x, state):
        out = F.relu(self.conv1(x))
        return out + self.shortcut(state)

        # return out


import os
import torch
from torch import nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        # self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        # self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    # def init_hidden(self, batch_size):
    #     return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next



############

class FlowBlock2DPropBegin(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bias=False):
        super().__init__()
        self.out_planes = in_planes-planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)


    def forward(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter)

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter
    
    def forward2(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter.detach())

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter


class FlowBlock2D3x3(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bias=False):
        super().__init__()
        self.out_planes = in_planes-planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)


    def forward(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter)

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter
    
    def forward2(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter.detach())

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter
        

class FlowBlock2D3x3Tanh(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bias=False):
        super().__init__()
        self.out_planes = in_planes-planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=bias)


    def forward(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter)

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter
    
    def forward2(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter.detach())

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.relu(out)

        return out, inter

class FlowBlock2DPropBeginTanh(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1,bias=False):
        super().__init__()
        self.out_planes = in_planes-planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=bias)


    def forward(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter)

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.tanh(out)

        return out, inter
    
    def forward2(self, x):
        inter = F.relu(self.conv1(x))
        inter = self.conv2(inter.detach())

        out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
        out = F.tanh(out)

        return out, inter
        

# class FlowBlock2DProp3Begin(nn.Module):
#     """Basic residual block class 2D"""

#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1,bias=False, return_inter=False):
#         super().__init__()
#         self.out_planes = planes
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=bias)

#         self.conv11 = nn.Conv2d(planes, planes, kernel_size=1,
#                                stride=1, padding=0, bias=bias)

#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=1,
#                                stride=1, padding=0, bias=bias)

#         self.return_inter = return_inter

#     def forward(self, x):
#         inter = F.relu(self.conv1(x))
#         inter = F.relu(self.conv11(x))
#         inter = self.conv2(inter)
#         out = x + torch.cat((inter, torch.zeros_like(x[:, :self.out_planes])), dim=1)
#         out = F.relu(out)

#         if self.return_inter:
#             return out, inter
#         else:
#             return out