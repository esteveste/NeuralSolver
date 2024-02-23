'''
FIXME import, convert name

dt_convlstm_ln -> neuralthink
'''

from typing import Optional, Tuple
import torch
from torch import Tensor, nn


from torch.autograd import Variable
# import numpy as np



import torch.nn.functional as F
# import torch.distributions as tdist
# # import numpy as np

# from functools import cached_property

import math
from typing import Iterable

import torch.nn as nn
from torch.nn import Parameter


class ConvLSTMCellV3(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias,
                dropout: float = 0.0,
                dropout_method: str = "pytorch",
                ln_preact: bool = True,
                learnable: bool = True,
                use_instance_norm=False,
                conv_dim: int = 2):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCellV3, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # group norm works for both 1d and 2d

        self.conv_dim = conv_dim

        if conv_dim == 2:
            conv_class = nn.Conv2d
            instance_norm_class = nn.InstanceNorm2d
        elif conv_dim == 1:
            conv_class = nn.Conv1d
            instance_norm_class = nn.InstanceNorm1d

            self.kernel_size = self.kernel_size[0]
            self.padding = self.padding[0]

        else:
            raise ValueError("conv_dim must be 1 or 2")

        self.conv_i2h = conv_class(in_channels=self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        self.conv_h2h = conv_class(in_channels=self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
        # 1,channels is equivalent to layernorm

        if use_instance_norm:
            if ln_preact:
                self.ln_i2h = instance_norm_class(4 * self.hidden_dim, affine=learnable)
                self.ln_h2h = instance_norm_class(4 * self.hidden_dim, affine=learnable)
            self.ln_cell = instance_norm_class(self.hidden_dim, affine=learnable)


        else:

            if ln_preact:
                self.ln_i2h = nn.GroupNorm(1,4 * self.hidden_dim, affine=learnable)
                self.ln_h2h = nn.GroupNorm(1,4 * self.hidden_dim, affine=learnable)
            self.ln_cell = nn.GroupNorm(1,self.hidden_dim, affine=learnable)
        self.ln_preact = ln_preact

        self.dropout = dropout
        self.dropout_method = dropout_method

        self.reset_parameters()

    def forward_input(self,input_tensor):
        i2h = self.conv_i2h(input_tensor)
        if self.ln_preact:
            i2h = self.ln_i2h(i2h)
        return i2h

    def forward(self, i2h, cur_state=None):
        do_dropout = self.training and self.dropout > 0.0


        if cur_state is None:
            if self.conv_dim == 2:
                cur_state = self.init_hidden(i2h.shape[0],(i2h.shape[2],i2h.shape[3]))
            elif self.conv_dim == 1:
                cur_state = self.init_hidden(i2h.shape[0],(i2h.shape[2],))
        
        h_cur, c_cur = cur_state

        # # Linear mappings
        # i2h = self.conv_i2h(input_tensor)
        h2h = self.conv_h2h(h_cur)
        if self.ln_preact:
            # i2h = self.ln_i2h(i2h)
            h2h = self.ln_h2h(h2h)
        combined_conv = i2h + h2h

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if do_dropout and self.dropout_method == "input":
            cc_i = F.dropout(cc_i, p=self.dropout, training=self.training)


        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f) # we could do bias unit init
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)


        # cell computations
        if do_dropout and self.dropout_method == "semeniuta":
            g = F.dropout(g, p=self.dropout, training=self.training)


        c_next = f * c_cur + i * g


        if do_dropout and self.dropout_method == "moon":
            ## be careful about shapes
            c_next.data.set_(torch.mul(c_next, self.mask).data)
            c_next.data *= 1.0 / (1.0 - self.dropout)

        c_next = self.ln_cell(c_next)

        h_next = o * torch.tanh(c_next)

        if do_dropout:
            if self.dropout_method == "pytorch":
                F.dropout(h_next, p=self.dropout, training=self.training, inplace=True)
            if self.dropout_method == "gal":
                h_next.data.set_(torch.mul(h_next, self.mask).data)
                h_next.data *= 1.0 / (1.0 - self.dropout)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        if self.conv_dim == 2:
            height, width = image_size
            return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_i2h.weight.device),
                    torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv_i2h.weight.device))
        elif self.conv_dim == 1:
            height = image_size[0]
            return (torch.zeros(batch_size, self.hidden_dim, height, device=self.conv_i2h.weight.device),
                    torch.zeros(batch_size, self.hidden_dim, height, device=self.conv_i2h.weight.device))
        else:
            raise ValueError("conv_dim must be 1 or 2")
    

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def sample_mask(self,device='cpu'):
        keep = 1.0 - self.dropout
        self.mask = torch.bernoulli(torch.empty(1, self.hidden_size, 1,1).fill_(keep)).to(device)


class GalDropout(nn.Module):
    """Applies GAL dropout to input samples with a fixed mask."""
    def __init__(self, dropout=0):
        super().__init__()

        assert 0 <= dropout < 1
        self._mask = None
        self._dropout = dropout

    def set_weights(self, X):
        """Calculates a new dropout mask."""

        mask = Variable(torch.ones_like(X), requires_grad=False)

        if X.is_cuda:
            mask = mask.cuda()

        ## also scales mask correctly, this is same as GAL dropout
        self._mask = F.dropout(mask, p=self._dropout, training=self.training)

    def forward(self, X):
        """Applies dropout to the input X."""
        if not self.training or not self._dropout:
            return X
        else:
            return X * self._mask

class NetConvLSTM_LN_Reduce(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, width, output_size, in_channels=3, recall=True, group_norm=False,bias=False,
                  _dropout=0,dropout_method='pytorch',ln_preact=True,use_instance_norm=False,
                  _dropout_gal2=0,norm_affine=True,
                  lstm_class=ConvLSTMCellV3,
                  conv_dim: int = 2,
                  use_pooling=True,
                  use_AvgPool=False,
                  use_smaller_head=False,**kwargs):
        super().__init__()

        self.name = "NetConvLSTM_LN_Reduce"

        self.bias = bias

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        self.use_pooling = use_pooling

        if conv_dim==2:
            conv_class = nn.Conv2d
        elif conv_dim==1:
            conv_class = nn.Conv1d
            in_channels=1 # yeah donno why... thanks original coders
        else:
            assert False, "not implemented"

        assert conv_dim==2, "not implemented conv1d"


        if conv_dim==2:
            if use_smaller_head:
                head_conv1 = conv_class(width, 32, kernel_size=3,
                    stride=1, padding=1, bias=bias)
                head_conv2 = conv_class(32, 8, kernel_size=3,
                                    stride=1, padding=1, bias=bias)
                head_conv3 = conv_class(8, output_size, kernel_size=3,
                                    stride=1, padding=1, bias=bias)

            else:

                head_conv1 = conv_class(width, width, kernel_size=3,
                                    stride=1, padding=1, bias=bias)
                head_conv2 = conv_class(width, width, kernel_size=3,
                                    stride=1, padding=1, bias=bias)
                head_conv3 = conv_class(width, output_size, kernel_size=3,
                                    stride=1, padding=1, bias=bias)
        elif conv_dim==1:
            head_conv1 = conv_class(width, width, kernel_size=3,
                                stride=1, padding=1, bias=bias)
            head_conv2 = conv_class(width, int(width/2), kernel_size=3,
                                stride=1, padding=1, bias=bias)
            head_conv3 = conv_class(int(width/2), 2, kernel_size=3,
                                stride=1, padding=1, bias=bias)

        else:
            assert False, "not implemented"

        if use_pooling:
            
            if use_AvgPool:
                head_pool = nn.AdaptiveAvgPool2d(output_size=1) 
            else:
                head_pool = nn.AdaptiveMaxPool2d(output_size=1)
        

            self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                    head_conv2, nn.ReLU(),
                                    head_conv3,
                                    
                                    head_pool,
                                    )
        else:
            self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                    head_conv2, nn.ReLU(),
                                    head_conv3,
                                    )

        assert dropout_method in ['pytorch','gal','moon','semeniuta','input']


        self.lstm = lstm_class(in_channels, width, (3,3), True,
                                   dropout=_dropout,ln_preact=ln_preact,dropout_method=dropout_method,
                                   use_instance_norm=use_instance_norm,learnable=norm_affine,
                                   conv_dim=conv_dim)

        self._dropout_h = _dropout_gal2
        self._state_drop = GalDropout(dropout=self._dropout_h)

        self.output_size = output_size

    def forward(self, x, iters_to_do, state=None, interim_thought=None, return_all_outputs=False, **kwargs):
        if interim_thought is not None:
            assert False, "not implemented - use lstm state instead"

        if self.use_pooling:
            all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)
        else:
            if len(x.shape)==4:
                all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2), x.size(3))).to(x.device)
            elif len(x.shape)==3:
                all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2))).to(x.device)
            else:
                assert False, "not implemented"

        mul=5
        self.lstm.sample_mask(x.device)

        lstm_inp1 = self.lstm.forward_input(x)

        for i in range(iters_to_do*mul):
            if i!=0:
                state = (interim_thought,c)
                

            interim_thought, c = self.lstm(lstm_inp1,state)
            if i==0:
                self._state_drop.set_weights(interim_thought)
            interim_thought = self._state_drop(interim_thought)


            if i%mul==mul-1:
                out = self.head(interim_thought).view(x.size(0), self.output_size)
                all_outputs[:, i//mul] = out


        if self.training:
            if return_all_outputs:
                return all_outputs, out, interim_thought
            else:
                return out, interim_thought

        return all_outputs


### definitions for paper

def neuralthink_1l_sgal04_py03_2d_out4_avgpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=True)

def neuralthink_1l_sgal04_py03_2d_out10_avgpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=10, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=True)


def neuralthink_1l_2d_out4_avgpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                    #  _dropout_gal2=0.4,
                     norm_affine=False,
                    # _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=True)

def neuralthink_1l_2d_out3_avgpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                    #  _dropout_gal2=0.4,
                     norm_affine=False,
                    # _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=True)

def neuralthink_1l_sgal04_py03_2d_out3_avgpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=True)

def neuralthink_1l_sgal04_py03_2d_out4_maxpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=False)

def neuralthink_1l_sgal04_py03_2d_out3_maxpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_AvgPool=False)

def neuralthink_1l_2d_out4_maxpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0,norm_affine=False,
                    _dropout=0,dropout_method='pytorch',
                    use_AvgPool=False)

def neuralthink_1l_2d_out3_maxpool(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0,norm_affine=False,
                    _dropout=0,dropout_method='pytorch',
                    use_AvgPool=False)

def neuralthink_1d(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    conv_dim=1,use_pooling=False,use_smaller_head=True)

def neuralthink_2d(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0.4,norm_affine=False,
                    _dropout=0.3,dropout_method='pytorch',
                    use_pooling=False,use_smaller_head=True)


def neuralthink_2d_nodrop(width, **kwargs):
    return NetConvLSTM_LN_Reduce(width=width, in_channels=kwargs["in_channels"], recall=True,
                     _dropout_gal2=0,norm_affine=False,
                    _dropout=0,dropout_method='pytorch',
                    use_pooling=False,use_smaller_head=True)

