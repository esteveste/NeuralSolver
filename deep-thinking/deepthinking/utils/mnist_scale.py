import json
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import PIL
from PIL import Image



def resize_grayscale(img, size):
    ## input (1,size,size) tensor
    ## output (1,size,size) tensor
    return F.interpolate(img[None], size=size)[0]

def place_digit_randomly(canvas, digit,max=None):
    x = np.random.randint(0,canvas.shape[1]-digit.shape[1]+1)
    y = np.random.randint(0,canvas.shape[2]-digit.shape[2]+1)
    canvas[:,x:x+digit.shape[1],y:y+digit.shape[2]] += digit

    # if max is not None:
    #     canvas.clamp_max_(max)

    return canvas

def place_digit_on_center(canvas, digit):
    x = (canvas.shape[1]-digit.shape[1])//2
    y = (canvas.shape[2]-digit.shape[2])//2
    canvas[:,x:x+digit.shape[1],y:y+digit.shape[2]] += digit
    return canvas

def crop_center(img,cropx,cropy):
    cropx = min(cropx, img.shape[1])
    cropy = min(cropy, img.shape[2])

    c, x, y = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,startx:startx+cropx,starty:starty+cropy]



class CenterScaleMnist(torch.utils.data.Dataset):

    ### same as MNIST Large Scale data set 
    ## isto tem o problema de ter de aprender a generalizar com crops nos digitos

    train_scales = [1,2,4]
    test_scales = [np.exp2(i/4) for i in range(-4,12+1)] #17 values, [0.5,8]
    def __init__(self, root: str, train: bool, download: bool = True, transform=None, resize_values: list = None, canvas_size: int = 112):

        self.root = root
        self.train = train
        self.transform = transform
        self.canvas_size = canvas_size if isinstance(canvas_size, tuple) or isinstance(canvas_size,list) else (canvas_size, canvas_size)

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download)

        self.inputs = self.mnist_dataset.data
        self.targets = self.mnist_dataset.targets

        if resize_values is not None:
            self.resize_values = resize_values
        else:
            get_resize_values = lambda l: [int(np.ceil(r*28)) for r in l]
            self.resize_values = get_resize_values(self.train_scales) if self.train else get_resize_values(self.test_scales)


    def __getitem__(self, index):

        canvas = torch.zeros(1,self.canvas_size[0],self.canvas_size[1])
        
        random_size = np.random.choice(self.resize_values)
        digit = resize_grayscale(self.inputs[index][None],random_size)


        if random_size > min(self.canvas_size):
            digit = crop_center(digit, self.canvas_size[0], self.canvas_size[1])

        canvas = place_digit_on_center(canvas, digit)

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

class ShiftScaleMnist(torch.utils.data.Dataset):

    ### same as MNIST Large Scale data set
    ## but with random shift

    train_scales = [1,2,4]
    test_scales = [np.exp2(i/4) for i in range(-4,12+1)] #17 values, [0.5,8]
    def __init__(self, root: str, train: bool, download: bool = True, transform=None, resize_values: list = None, canvas_size: int = 112):

        self.root = root
        self.train = train
        self.transform = transform
        self.canvas_size = canvas_size if isinstance(canvas_size, tuple) or isinstance(canvas_size,list) else (canvas_size, canvas_size)

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download)

        self.inputs = self.mnist_dataset.data
        self.targets = self.mnist_dataset.targets

        if resize_values is not None:
            self.resize_values = resize_values 
        else:
            get_resize_values = lambda l: [int(np.ceil(r*28)) for r in l]
            self.resize_values = get_resize_values(self.train_scales) if self.train else get_resize_values(self.test_scales)


    def __getitem__(self, index):

        canvas = torch.zeros(1,self.canvas_size[0],self.canvas_size[1])
        
        random_size = np.random.choice(self.resize_values)
        digit = resize_grayscale(self.inputs[index][None],random_size)

        if random_size > min(self.canvas_size):
            digit = crop_center(digit, self.canvas_size[0], self.canvas_size[1])

        canvas = place_digit_randomly(canvas, digit)

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class ShiftConstantMnist(torch.utils.data.Dataset):


    # train_scales = [1,2,4]
    # test_scales = [np.exp2(i/4) for i in range(-4,12+1)] #17 values, [0.5,8]

    # base_folder = "prefix_sums_data"
    # # url = "https://cs.umd.edu/~tomg/download/Easy_to_Hard_Datav2/prefix_sums_data.tar.gz"
    # lengths = list(range(16, 65)) + [72] + [128] + [256] + [512]
    # download_list = [f"mod3_{l}_data.pth" for l in lengths] + [f"mod3_{l}_targets.pth" for l in lengths]

    def __init__(self, root: str, train: bool, canvas_size, download: bool = True, transform=None):

        self.root = root
        self.train = train
        self.transform = transform
        self.canvas_size = canvas_size if isinstance(canvas_size, tuple) or isinstance(canvas_size,list) else (canvas_size, canvas_size)

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download)

        self.inputs = self.mnist_dataset.data
        self.targets = self.mnist_dataset.targets


    def __getitem__(self, index):
        canvas = torch.zeros(1, self.canvas_size[0], self.canvas_size[1])
        canvas = place_digit_randomly(canvas, self.inputs[index][None])

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.targets[index]

    def __len__(self):
        return self.inputs.size(0)


class CenterScaleMnist_v2(torch.utils.data.Dataset):

    ### all possible scales
    ### no crop of digit

    def __init__(self, root: str, train: bool, download: bool = True, transform=None, canvas_size: int = 112):

        self.root = root
        self.train = train
        self.transform = transform
        self.canvas_size = canvas_size if isinstance(canvas_size, tuple) or isinstance(canvas_size,list) else (canvas_size, canvas_size)

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download)

        self.inputs = self.mnist_dataset.data
        self.targets = self.mnist_dataset.targets


    def __getitem__(self, index):

        canvas = torch.zeros(1,self.canvas_size[0],self.canvas_size[1])
        
        random_size = np.random.randint(28, min(self.canvas_size))
        digit = resize_grayscale(self.inputs[index][None],random_size)


        if random_size > min(self.canvas_size):
            digit = crop_center(digit, self.canvas_size[0], self.canvas_size[1])

        canvas = place_digit_on_center(canvas, digit)

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.targets[index]

    def __len__(self):
        return self.inputs.size(0)
class ShiftScaleMnist_v2(torch.utils.data.Dataset):

    ### all possible scales, no crop of digit
    ## with random shift
    
    def __init__(self, root: str, train: bool, download: bool = True, transform=None, canvas_size: int = 112):

        self.root = root
        self.train = train
        self.transform = transform
        self.canvas_size = canvas_size if isinstance(canvas_size, tuple) or isinstance(canvas_size,list) else (canvas_size, canvas_size)

        self.mnist_dataset = datasets.MNIST(root, train=train, download=download)

        self.inputs = self.mnist_dataset.data
        self.targets = self.mnist_dataset.targets


    def __getitem__(self, index):

        canvas = torch.zeros(1,self.canvas_size[0],self.canvas_size[1])

        random_size = np.random.randint(28, min(self.canvas_size))
        digit = resize_grayscale(self.inputs[index][None],random_size)

        if random_size > min(self.canvas_size):
            digit = crop_center(digit, self.canvas_size[0], self.canvas_size[1])

        canvas = place_digit_randomly(canvas, digit)

        if self.transform is not None:
            canvas = self.transform(canvas)

        return canvas, self.targets[index]

    def __len__(self):
        return self.inputs.size(0)

import torch
from torch.utils import data
# from easy_to_hard_data import *

from math import ceil, floor

def prepare_mnist_loader(problem_type, train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, shuffle=True):

    if problem_type == "shift_constant_mnist":
        dataset_class = ShiftConstantMnist
    elif problem_type == "shift_scale_mnist":
        dataset_class = ShiftScaleMnist_v2
    elif problem_type == "center_scale_mnist":
        dataset_class = CenterScaleMnist_v2

    # elif problem_type == "prefix_sums_last2":
    #     dataset_class = PrefixSumLast2Dataset
    
    else:
        raise ValueError(f"problem_type unknown {problem_type}")

    transform = transform = lambda x: x/255


    dataset = dataset_class("../../../data", canvas_size=train_data, transform=transform, train=True)
    testset = dataset_class("../../../data", canvas_size=test_data, transform=transform, train=False)

    train_split = int(train_split * len(dataset))

    trainset, valset = torch.utils.data.random_split(dataset,
                                                     [train_split,
                                                      int(len(dataset) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                  shuffle=shuffle, drop_last=True)
    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    valloader = data.DataLoader(valset, num_workers=0, batch_size=test_batch_size,
                                shuffle=False, drop_last=False)
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


### examples

# ### create centerd dataset
# train_dataset = ShiftScaleMnist_v2('../data', train=True, download=True, canvas_size=64)
# test_dataset = ShiftScaleMnist_v2('../data', train=False, download=True, canvas_size=512)

# train_dataset = CenterScaleMnist_v2('../data', train=True, download=True, canvas_size=112)
# test_dataset = CenterScaleMnist_v2('../data', train=False, download=True, canvas_size=512)


# # ### create centerd dataset
# train_dataset = ShiftConstantMnist('../data', canvas_size=40, train=True, download=True)
# test_dataset = ShiftConstantMnist('../data', canvas_size=[100,200], train=False, download=True)


# ### create dataloader
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
