from typing import Optional, Tuple
import torch
from torch import Tensor, nn

from .blocks import BasicBlock2D as BasicBlock
from .blocks import BasicBlock2DNoFinalRelu
from .blocks import *

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, in_channels=3, recall=True, group_norm=False,bias=False, **kwargs):
        super().__init__()

        self.bias = bias

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=bias)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=bias)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=bias)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)




    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm, bias=self.bias))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, return_all_outputs=False, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            if return_all_outputs:
                return all_outputs, out, interim_thought
            else:
                return out, interim_thought

        return all_outputs



def dt_net_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False)


def dt_net_recall_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True)


def dt_net_recall_2d_with_bias(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True,bias=True)


def dt_net_recall_2d_1block(width, **kwargs):
    return DTNet(BasicBlock, [1], width=width, in_channels=kwargs["in_channels"], recall=True)

def dt_net_recall_2d_1block_noprop(width, **kwargs):
    return DTNet(BasicBlock2DNoProp, [1], width=width, in_channels=kwargs["in_channels"], recall=True)

def dt_net_recall_2d_onlyrecall(width, **kwargs):
    return DTNet(BasicBlock2DNoProp, [], width=width, in_channels=kwargs["in_channels"], recall=True)

def dt_net_recall_2d_2block1x1(width, **kwargs):
    return DTNet(BasicBlock2D1x1, [2], width=width, in_channels=kwargs["in_channels"], recall=True)




def dt_net_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=False, group_norm=True)


def dt_net_recall_gn_2d(width, **kwargs):
    return DTNet(BasicBlock, [2], width=width, in_channels=kwargs["in_channels"], recall=True, group_norm=True)


class DTNetReduceMaxPool(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, output_size, in_channels=3, recall=True, group_norm=False, use_AvgPool=False,**kwargs):
        super().__init__()

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        if use_AvgPool:
            head_pool = nn.AdaptiveAvgPool2d(output_size=1)
        else:
            head_pool = nn.AdaptiveMaxPool2d(output_size=1)

        head_conv1 = nn.Conv2d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(32, output_size, kernel_size=3,
                               stride=1, padding=1, bias=False)

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
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, return_all_outputs=False,  **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought).view(x.size(0), self.output_size)

            all_outputs[:, i] = out


        if self.training:
            if return_all_outputs:
                return all_outputs, out, interim_thought
            else:
                return out, interim_thought

        return all_outputs

def dt_net_recall_2d_out10(width, **kwargs):
    return DTNetReduceMaxPool(BasicBlock, [2], width=width, output_size=10, in_channels=kwargs["in_channels"], recall=True)

def dt_net_recall_2d_out4(width, **kwargs):
    return DTNetReduceMaxPool(BasicBlock, [2], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=True)

def dt_net_2d_out4(width, **kwargs):
    return DTNetReduceMaxPool(BasicBlock, [2], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=False)

def dt_net_2d_out4_avg(width, **kwargs):
    return DTNetReduceMaxPool(BasicBlock, [2], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=False,use_AvgPool=True)

def dt_net_recall_2d_out4_1block(width, **kwargs):
    return DTNetReduceMaxPool(BasicBlock, [1], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=True)



# class DTNetReduceMaxPoolEnd(nn.Module):
#     """DeepThinking Network 2D model class"""

#     def __init__(self, block, num_blocks, width, output_size, in_channels=3, recall=True, group_norm=False, use_AvgPool=False,**kwargs):
#         super().__init__()

#         self.recall = recall
#         self.width = int(width)
#         self.group_norm = group_norm
#         proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
#                               stride=1, padding=1, bias=False)

#         conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
#                                 stride=1, padding=1, bias=False)

#         recur_layers = []
#         if recall:
#             recur_layers.append(conv_recall)

#         for i in range(len(num_blocks)):
#             recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

#         if use_AvgPool:
#             head_pool = nn.AdaptiveAvgPool2d(output_size=1)
#         else:
#             head_pool = nn.AdaptiveMaxPool2d(output_size=1)

#         head_conv1 = nn.Conv2d(width, width, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         head_conv2 = nn.Conv2d(width, 32, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         head_conv3 = nn.Conv2d(32, output_size, kernel_size=3,
#                                stride=1, padding=1, bias=False)

#         self.projection = nn.Sequential(proj_conv, nn.ReLU())
#         self.recur_block = nn.Sequential(*recur_layers)
#         self.head = nn.Sequential(head_conv1, nn.ReLU(),
#                                   head_conv2, nn.ReLU(),
#                                   head_conv3,
#                                   head_pool)

#         self.output_size = output_size

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for strd in strides:
#             layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
#             self.width = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x, iters_to_do, interim_thought=None, return_all_outputs=False,  **kwargs):
#         initial_thought = self.projection(x)

#         if interim_thought is None:
#             interim_thought = initial_thought

#         all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)

#         for i in range(iters_to_do):
#             if self.recall:
#                 interim_thought = torch.cat([interim_thought, x], 1)
#             interim_thought = self.recur_block(interim_thought)
#             out = self.head(interim_thought).view(x.size(0), self.output_size)

#             all_outputs[:, i] = out

#         if self.training:
#             if return_all_outputs:
#                 return all_outputs, out, interim_thought
#             else:
#                 return out, interim_thought

#         return all_outputs

# def dt_net_recall_2d_out4_end(width, **kwargs):
#     return DTNetReduceMaxPoolEnd(BasicBlock, [2], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=True)

# def dt_net_recall_2d_out4_avg_end(width, **kwargs):
#     return DTNetReduceMaxPoolEnd(BasicBlock, [2], width=width, output_size=4, in_channels=kwargs["in_channels"], recall=True,use_AvgPool=True)


# def dt_net_recall_2d_out10_end(width, **kwargs):
#     return DTNetReduceMaxPoolEnd(BasicBlock, [2], width=width, output_size=10, in_channels=kwargs["in_channels"], recall=True)



class FeedForwardNetAvgPoolEnd(nn.Module):
    """Modified Residual Network model class"""

    def __init__(self, block, num_blocks, width, output_size, in_channels=3, recall=True, max_iters=8, group_norm=False,
                 use_AvgPool=True, paper_out_head=False,):
        super().__init__()

        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)

        if self.recall:
            self.recall_layer = nn.Conv2d(width+in_channels, width, kernel_size=3,
                                          stride=1, padding=1, bias=False)
        else:
            self.recall_layer = nn.Sequential()

        self.feedforward_layers = nn.ModuleList()
        for _ in range(max_iters):
            internal_block = []
            for j in range(len(num_blocks)):
                internal_block.append(self._make_layer(block, width, num_blocks[j], stride=1))
            self.feedforward_layers.append(nn.Sequential(*internal_block))


        if use_AvgPool:
            head_pool = nn.AdaptiveAvgPool2d(output_size=1) 
            # if we do average we are dividing all outputs by the size of the image.
        else:
            head_pool = nn.AdaptiveMaxPool2d(output_size=1)
        
        head_conv1 = nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(32, output_size, kernel_size=3, stride=1, padding=1, bias=False)

        if paper_out_head:
            head_conv1 = nn.Conv2d(width, 64, kernel_size=3, stride=1, padding=1, bias=False)
            head_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            head_conv3 = nn.Conv2d(64, output_size, kernel_size=3, stride=1, padding=1, bias=False)


        self.iters = max_iters
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.head = nn.Sequential(head_conv1, nn.ReLU(), head_conv2, nn.ReLU(), head_conv3, head_pool)

        self.output_size = output_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, iters_elapsed=0, **kwargs):
        # assert (iters_elapsed + iters_to_do) <= self.iters
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)

        for i, layer in enumerate(self.feedforward_layers[iters_elapsed:iters_elapsed+iters_to_do]):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
                interim_thought = self.recall_layer(interim_thought)
            interim_thought = layer(interim_thought)
            out = self.head(interim_thought).view(x.size(0), self.output_size)
            all_outputs[:, i] = out

        if iters_to_do > self.iters:
            # fill in the rest with the last output
            all_outputs[:, self.iters:] = out.unsqueeze(1).repeat(1, iters_to_do - self.iters, 1)

        if self.training:
            return out, interim_thought
        else:
            return all_outputs


def feedforward_net_recall_2d_out4_avgpool(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=4, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"])


def feedforward_net_recall_2d_out3_avgpool(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=3, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"])

def feedforward_net_recall_2d_out4_maxpool(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=4, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"],use_AvgPool=False)

def feedforward_net_recall_2d_out3_maxpool(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=3, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"],use_AvgPool=False)



def feedforward_net_recall_2d_out4_maxpool_fixhead(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=4, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"],use_AvgPool=False,
                          paper_out_head=True)


def feedforward_net_recall_2d_out3_maxpool_fixhead(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=3, in_channels=kwargs["in_channels"],
                          recall=True, max_iters=kwargs["max_iters"],use_AvgPool=False,paper_out_head=True)



def feedforward_net_2d_out3_avgpool(width, **kwargs):
    return FeedForwardNetAvgPoolEnd(BasicBlock, [2], width, output_size=3, in_channels=kwargs["in_channels"],
                          recall=False, max_iters=kwargs["max_iters"])



class DTNetPool(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, output_size, in_channels=3, recall=True, group_norm=False,bias=False,
        use_AvgPool=True,paper_out_head=False,
     **kwargs):
        super().__init__()

        self.bias = bias

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=bias)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=bias)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv3 = nn.Conv2d(8, output_size, kernel_size=3,
                               stride=1, padding=1, bias=bias)


        if paper_out_head:
            head_conv1 = nn.Conv2d(width, 64, kernel_size=3, stride=1, padding=1, bias=False)
            head_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
            head_conv3 = nn.Conv2d(64, output_size, kernel_size=3, stride=1, padding=1, bias=False)



        if use_AvgPool:
            head_pool = nn.AdaptiveAvgPool2d(output_size=1) 
            # if we do average we are dividing all outputs by the size of the image.
        else:
            head_pool = nn.AdaptiveMaxPool2d(output_size=1)
        

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3,head_pool)


        self.output_size = output_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm, bias=self.bias))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, return_all_outputs=False, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)

            # out = self.head(interim_thought)
            out = self.head(interim_thought).view(x.size(0), self.output_size)
            all_outputs[:, i] = out

        if self.training:
            if return_all_outputs:
                return all_outputs, out, interim_thought
            else:
                return out, interim_thought

        return all_outputs


def dt_net_recall_2d_out3_avg_pool(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True)


def dt_net_recall_2d_out4_avg_pool(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True)

def dt_net_recall_2d_out3_maxpool(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                        use_AvgPool=False)

def dt_net_recall_2d_out3_maxpool_fixhead(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True,
                        use_AvgPool=False,paper_out_head=True)


def dt_net_recall_2d_out4_maxpool(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                        use_AvgPool=False)


def dt_net_recall_2d_out4_maxpool_fixhead(width, **kwargs):
    return DTNetPool(BasicBlock, [2], width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True,
                        use_AvgPool=False,paper_out_head=True)

class DTNetRandom(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, block, num_blocks, width, output_size, in_channels=3, recall=True, group_norm=False,bias=False, **kwargs):
        super().__init__()

        self.bias = bias

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=bias)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=bias)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=bias)
        head_conv3 = nn.Conv2d(8, output_size, kernel_size=3,
                               stride=1, padding=1, bias=bias)

        head_pool = nn.AdaptiveAvgPool2d(output_size=1) 

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3,head_pool)


        self.output_size = output_size

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm, bias=self.bias))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, return_all_outputs=False, **kwargs):
       

        all_outputs = torch.rand((x.size(0), iters_to_do, self.output_size),requires_grad=True).to(x.device)

        # for i in range(iters_to_do):
        #     if self.recall:
        #         interim_thought = torch.cat([interim_thought, x], 1)
        #     interim_thought = self.recur_block(interim_thought)

        #     # out = self.head(interim_thought)
        #     out = self.head(interim_thought).view(x.size(0), self.output_size)
        #     all_outputs[:, i] = out

        if self.training:
            out = all_outputs[:, -1]
            if return_all_outputs:
                return all_outputs, out, interim_thought
            else:
                return out, interim_thought

        return all_outputs


def dt_net_random_out3(width, **kwargs):
    return DTNetRandom(BasicBlock, [2], width=width,output_size=3, in_channels=kwargs["in_channels"], recall=True)

def dt_net_random_out4(width, **kwargs):
    return DTNetRandom(BasicBlock, [2], width=width,output_size=4, in_channels=kwargs["in_channels"], recall=True)
