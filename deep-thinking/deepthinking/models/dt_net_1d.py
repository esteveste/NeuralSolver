
import torch
from torch import nn

from .blocks import BasicBlock1D as BasicBlock
from .blocks import BasicBlock1DBN as BasicBlockBN

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet1D(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, block, num_blocks, width, recall, group_norm=False, **kwargs):
        super().__init__()

        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        proj_conv = nn.Conv1d(1, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv1d(width + 1, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        if self.recall:
            recur_layers = [conv_recall, nn.ReLU()]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv1d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv1d(width, int(width/2), kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv1d(int(width/2), 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)

            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs


def dt_net_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=False)


def dt_net_recall_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=True)


def dt_net_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=False, group_norm=True)


def dt_net_recall_gn_1d(width, **kwargs):
    return DTNet1D(BasicBlock, [2], width, recall=True, group_norm=True)

def dt_net_recall_bn_1d(width, **kwargs):
    return DTNet1D(BasicBlockBN, [2], width, recall=True, group_norm=True)



class DTNet1DReduceMaxPool(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, block, num_blocks, width, recall, output_size, group_norm=False, **kwargs):
        super().__init__()

        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        proj_conv = nn.Conv1d(1, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv1d(width + 1, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        if self.recall:
            recur_layers = [conv_recall, nn.ReLU()]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv1d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False) # same width calculation

        # head_pool= nn.AdaptiveAvgPool1d(output_size=1) ## COULD HAVE VERY BIG SUMS

        head_pool = nn.AdaptiveMaxPool1d(output_size=1)

        head_conv2 = nn.Conv1d(width, width, kernel_size=1,
                               stride=1, padding=0, bias=False)


        head_conv3 = nn.Conv1d(width, 2*output_size, kernel_size=1,
                               stride=1, padding=0, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_pool,
                                  head_conv2, nn.ReLU(),
                                  head_conv3)
        
        self.output_size = output_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, self.output_size)).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)

            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought).view(x.size(0), 2, self.output_size)


            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought

        return all_outputs
    


def dt_net_1d_out2(width, **kwargs):
    return DTNet1DReduceMaxPool(BasicBlock, [2], width, output_size=2, recall=False)


def dt_net_recall_1d_out2(width, **kwargs):
    return DTNet1DReduceMaxPool(BasicBlock, [2], width, output_size=2, recall=True)

def dt_net_recall_1d_out4(width, **kwargs):
    return DTNet1DReduceMaxPool(BasicBlock, [2], width, output_size=4, recall=True)

def dt_net_recall_1d_out8(width, **kwargs):
    return DTNet1DReduceMaxPool(BasicBlock, [2], width, output_size=8, recall=True)
