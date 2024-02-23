
from dataclasses import dataclass
from random import randrange

import numpy as np
import torch
from icecream import ic
from tqdm import tqdm

from deepthinking.utils.testing import get_predicted

import wandb

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114, W0611


@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "typing.Any"
    scheduler: "typing.Any"
    warmup: "typing.Any"
    clip: "typing.Any"
    alpha: "typing.Any"
    max_iters: "typing.Any"
    problem: "typing.Any"

    ## MINE
    custom_beta: "typing.Any"
    noise_l1_alpha: "typing.Any"
    noise_l1_only_final_and_next: "typing.Any"
    min_iters: "typing.Any"


def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k


def get_output_for_prog_loss_add_noise(inputs, max_iters, net,custom_beta):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n, alpha_noise=custom_beta)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought,alpha_noise=custom_beta)
    return outputs, k

def get_output_for_random_pert_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought
        # random perturbation
        interim_thought = interim_thought + torch.randn_like(interim_thought) * interim_thought * 0.01
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def get_output_for_more_iters_loss(inputs, interim_thought, max_iters, net):
    # do k iterations using intermediate features as input
    k = randrange(1, max_iters)

    ## why detach? I think no difference computation wise
    # interim_thought = interim_thought.detach() 

    outputs, _ = net(inputs, iters_elapsed=max_iters, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def get_output_for_more_iters_loss_flow(inputs, interim_thought, max_iters, net):
    # do k iterations using intermediate features as input
    k = randrange(1, max_iters)

    ## why detach? I think no difference computation wise
    # interim_thought = interim_thought.detach() 

    outputs, _,recur_inter = net(inputs, iters_elapsed=max_iters, iters_to_do=k, interim_thought=interim_thought,return_recur_inter=True)
    return outputs, k,recur_inter

def train(net, loaders, mode, train_setup, device):
    if mode == "progressive":
        train_loss, acc = train_progressive(net, loaders, train_setup, device)

    elif mode == "progressive_loss_out":
        train_loss, acc = train_progressive_loss_out(net, loaders, train_setup, device)

    elif mode == "progressive_l1mask":
        train_loss, acc = train_progressive_l1mask(net, loaders, train_setup, device)

    elif mode == "progressive_flowl1":
        train_loss, acc = train_progressive_flowl1(net, loaders, train_setup, device)

    elif mode == "progressive_noise":
        train_loss, acc = train_progressive_noise(net, loaders, train_setup, device)


    elif mode == "random_iter_interval":
        train_loss, acc = train_random_iter_interval(net, loaders, train_setup, device)
    

    elif mode == "all_outputs":
        train_loss, acc = train_all_outputs(net, loaders, train_setup, device)    
    

    elif mode == "all_outputs_noiseL1":
        train_loss, acc = train_all_outputs_noiseL1(net, loaders, train_setup, device)    
    

    elif mode == "all_outputs_penalized":
        train_loss, acc = train_all_outputs_penalized(net, loaders, train_setup, device)    
    
    elif mode == "rl_distilation":
        train_loss, acc = train_rl_distilation(net, loaders, train_setup, device)
    # elif mode == "random_pertubation":
    #     train_loss, acc = train_random_pertubation(net, loaders, train_setup, device)    
      
    elif mode == "more_iter_after":
        train_loss, acc = train_more_iter_after(net, loaders, train_setup, device)    
    
    
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return train_loss, acc


def train_progressive(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        if alpha != 1:
            outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
            if alpha==1:
                outputs_max_iters = outputs
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        
        loss.backward()

        if clip is not None:
            n = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if batch_idx % 100 == 0:
                print(f"clipped grad norm: {n}")

        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    # do in training.py
    # lr_scheduler.step()
    # warmup_scheduler.dampen()

    return train_loss, acc




def train_progressive_loss_out(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    custom_beta = train_setup.custom_beta

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        if alpha != 1:
            outputs_max_iters, _, loss_out = net(inputs, iters_to_do=max_iters)
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            raise NotImplementedError("alpha != 0 not implemented for progressive_l1_out")
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
            if alpha==1:
                outputs_max_iters = outputs
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean + custom_beta * loss_out
        
        loss.backward()

        if clip is not None:
            n = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if batch_idx % 100 == 0:
                print(f"clipped grad norm: {n}")
                print(f"loss_out: {loss_out}")


        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    # do in training.py
    # lr_scheduler.step()
    # warmup_scheduler.dampen()

    return train_loss, acc

## from https://github.com/JoonyoungYi/KD-pytorch
# loss = criterion(outputs, pseudo_targets, targets)

from torch.nn import functional as F
import torch.nn as nn

def _make_criterion(alpha=0.5, T=4.0, mode='cse'):
    def criterion(outputs, targets, labels):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        _hard_loss = F.cross_entropy(outputs, labels)
        loss = alpha * _soft_loss + (1. - alpha) * _hard_loss
        return loss

    return criterion

def _make_soft_loss(T=4.0, mode='cse'):
    def soft_loss(outputs, targets):
        if mode == 'cse':
            _p = F.log_softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = -torch.mean(torch.sum(_q * _p, dim=1))
        elif mode == 'mse':
            _p = F.softmax(outputs / T, dim=1)
            _q = F.softmax(targets / T, dim=1)
            _soft_loss = nn.MSELoss()(_p, _q) / 2
        else:
            raise NotImplementedError()

        _soft_loss = _soft_loss * T * T
        return _soft_loss

    return soft_loss


def train_rl_distilation(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    # criterion = torch.nn.CrossEntropyLoss(reduction="none")
    # criterion = _make_soft_loss(T=4.0, mode='cse')
    criterion = _make_soft_loss(T=1.0, mode='cse')

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc

def train_more_iter_after(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, max_iter_thought = net(inputs, iters_to_do=max_iters)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_more_iters_loss(inputs, max_iter_thought, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc

# def train_random_pertubation(net, loaders, train_setup, device):
#     trainloader = loaders["train"]
#     net.train()
#     optimizer = train_setup.optimizer
#     lr_scheduler = train_setup.scheduler
#     warmup_scheduler = train_setup.warmup
#     alpha = train_setup.alpha
#     max_iters = train_setup.max_iters
#     k = 0
#     problem = train_setup.problem
#     clip = train_setup.clip
#     criterion = torch.nn.CrossEntropyLoss(reduction="none")

#     train_loss = 0
#     correct = 0
#     total = 0

#     for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
#         inputs, targets = inputs.to(device), targets.to(device).long()
#         targets = targets.view(targets.size(0), -1)
#         if problem == "mazes" or "mask" in problem:
#             mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

#         optimizer.zero_grad()

#         # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
#         # so we save time by settign it equal to 0).
#         outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)
#         if alpha != 1:
#             outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
#                                                        outputs_max_iters.size(1), -1)
#             loss_max_iters = criterion(outputs_max_iters, targets)
#         else:
#             loss_max_iters = torch.zeros_like(targets).float()

#         # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
#         # so we save time by setting it equal to 0).
#         if alpha != 0:
#             outputs, k = get_output_for_random_pert_loss(inputs, max_iters, net)
#             outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
#             loss_progressive = criterion(outputs, targets)
#         else:
#             loss_progressive = torch.zeros_like(targets).float()

#         if problem == "mazes" or "mask" in problem:
#             loss_max_iters = (loss_max_iters * mask)
#             loss_max_iters = loss_max_iters[mask > 0]
#             loss_progressive = (loss_progressive * mask)
#             loss_progressive = loss_progressive[mask > 0]

#         loss_max_iters_mean = loss_max_iters.mean()
#         loss_progressive_mean = loss_progressive.mean()

#         loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
#         loss.backward()

#         if clip is not None:
#             torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
#         optimizer.step()

#         train_loss += loss.item()
#         predicted = get_predicted(inputs, outputs_max_iters, problem)
#         correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
#         total += targets.size(0)

#     train_loss = train_loss / (batch_idx + 1)
#     acc = 100.0 * correct / total

#     lr_scheduler.step()
#     warmup_scheduler.dampen()

#     return train_loss, acc



def train_random_iter_interval(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    min_iters = train_setup.min_iters

    assert alpha == 0, "alpha must be 0 for this training method, since other is meaningless"

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        iter_todo = np.random.randint(min_iters, max_iters+1)

        outputs_max_iters, _ = net(inputs, iters_to_do=iter_todo)
        
        # if alpha != 1:

        outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                    outputs_max_iters.size(1), -1)
        loss_max_iters = criterion(outputs_max_iters, targets)
        # else:
        #     loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # # so we save time by setting it equal to 0).
        # if alpha != 0:
        #     outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
        #     outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        #     loss_progressive = criterion(outputs, targets)
        # else:
        
        # loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            # loss_progressive = (loss_progressive * mask)
            # loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        # loss_progressive_mean = loss_progressive.mean()

        # loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        
        loss = loss_max_iters_mean
        loss.backward()

        if clip is not None:
            n = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if batch_idx % 100 == 0:
                print(f"clipped grad norm: {n}")
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc

def train_all_outputs(net, loaders, train_setup, device):
    ## 1 all outputs
    ## 0 only last output


    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    min_iters = train_setup.min_iters-1
    assert train_setup.min_iters>0, "min_iters must be > 0, this is expected"

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).

        ## FIXME this is the out... not all outputs... last iteration out
        ## shape (B, 2, h, w) ...., all outputs -> (B, max_iters, 2, h, w)
        all_outputs, outputs_max_iters, _ = net(inputs, iters_to_do=max_iters, return_all_outputs=True)
        outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
        if alpha != 1:
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            ## B, iters, H*W
            all_targets = targets.unsqueeze(1).repeat(1, max_iters-min_iters, 1)
            ## B * iters, H*W
            all_targets = all_targets.view(all_targets.size(0)*all_targets.size(1), -1)

            ## B, iters, C, H*W
            outputs = all_outputs[:,min_iters:].reshape(all_outputs.size(0)*(all_outputs.size(1)-min_iters),all_outputs.size(2), -1)
            ## -> B*iters, C, H*W

            loss_all_loss = criterion(outputs, all_targets)
        else:
            loss_all_loss = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            ## B, C/cross, H, W -> B, H*W
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            ## ainda nao gosto disto, nao compreendo para que enforcar isto
            ### B,C, H*W  | B, C, H*W
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]

            ## B*iters,C, H*W
            mask_all = mask.unsqueeze(1).repeat(1, max_iters, 1)
            mask_all = mask_all.view(mask_all.size(0)*mask_all.size(1), -1)

            loss_all_loss = (loss_all_loss * mask_all)
            loss_all_loss = loss_all_loss[mask_all > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_all_loss_mean = loss_all_loss.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_all_loss_mean
        loss.backward()

        if clip is not None:
            n = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if batch_idx % 100 == 0:
                print(f"clipped grad norm: {n}")
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc



## fix from previous version
def train_all_outputs_noiseL1(net, loaders, train_setup, device):
    ## 1 all outputs
    ## 0 only last output


    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")


    noise_l1_alpha = train_setup.noise_l1_alpha
    noise_l1_only_final_and_next = train_setup.noise_l1_only_final_and_next

    custom_beta = train_setup.custom_beta

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).

        ## FIXME this is the out... not all outputs... last iteration out
        ## shape (B, 2, h, w) ...., all outputs -> (B, max_iters, 2, h, w)
        all_outputs, outputs_max_iters, _, noise_loss = net.forward_L1_interim(inputs, iters_to_do=max_iters, return_all_outputs=True, only_final_and_next=noise_l1_only_final_and_next,custom_beta=custom_beta)
        outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
        if alpha != 1:
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            ## B, iters, H*W
            all_targets = targets.unsqueeze(1).repeat(1, max_iters, 1)
            ## B * iters, H*W
            all_targets = all_targets.view(all_targets.size(0)*all_targets.size(1), -1)

            ## B, iters, C, H*W
            outputs = all_outputs.view(all_outputs.size(0), all_outputs.size(1),all_outputs.size(2), -1)
            ## B*iters, C, H*W
            outputs = outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2), -1)

            # outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_all_loss = criterion(outputs, all_targets)
        else:
            loss_all_loss = torch.zeros_like(targets).repeat(max_iters,1).float()

        if problem == "mazes" or "mask" in problem:
            ## B, C/cross, H, W -> B, H*W
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            ## ainda nao gosto disto, nao compreendo para que enforcar isto
            ### B,C, H*W  | B, C, H*W
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]

            ## B*iters,C, H*W
            mask_all = mask.unsqueeze(1).repeat(1, max_iters, 1)
            mask_all = mask_all.view(mask_all.size(0)*mask_all.size(1), -1)

            loss_all_loss = (loss_all_loss * mask_all)
            loss_all_loss = loss_all_loss[mask_all > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_all_loss_mean = loss_all_loss.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_all_loss_mean + noise_l1_alpha*noise_loss
        loss.backward()

        if clip is not None:
            n = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            if batch_idx % 100 == 0:
                print(f"clipped grad norm: {n}")
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()


    ## DEBUG, screws step....
    wandb.log({"noise_loss": noise_loss.item()})
    print(f"noise_loss: {noise_loss.item()}")


    return train_loss, acc


def train_all_outputs_penalized(net, loaders, train_setup, device):
    ## 1 all outputs
    ## 0 only last output


    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    linear_sum = max_iters/2*(max_iters+1)
    linear_terms = (torch.arange(1,max_iters+1).float().to(device)/linear_sum).reshape(1, max_iters, 1)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).

        ## FIXME this is the out... not all outputs... last iteration out
        ## shape (B, 2, h, w) ...., all outputs -> (B, max_iters, 2, h, w)
        all_outputs, outputs_max_iters, _ = net(inputs, iters_to_do=max_iters, return_all_outputs=True)
        outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
        if alpha != 1:
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:

            batch =targets.size(0)

            ## B, iters, H*W
            all_targets = targets.unsqueeze(1).repeat(1, max_iters, 1)
            
            ## B * iters, H*W
            all_targets = all_targets.view(all_targets.size(0)*all_targets.size(1), -1)

            ## B, iters, C, H*W
            outputs = all_outputs.view(all_outputs.size(0), all_outputs.size(1),all_outputs.size(2), -1)
            ## B*iters, C, H*W
            outputs = outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2), -1)

            # outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_all_loss = criterion(outputs, all_targets) # reduction already none here




        else:
            loss_all_loss = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            ## B, C/cross, H, W -> B, H*W
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

            ## ainda nao gosto disto, nao compreendo para que enforcar isto
            ### B,C, H*W  | B, C, H*W
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]

            ## B*iters,C, H*W
            mask_all = mask.unsqueeze(1).repeat(1, max_iters, 1)
            mask_all = mask_all.view(mask_all.size(0)*mask_all.size(1), -1)

            loss_all_loss = (loss_all_loss * mask_all)
            loss_all_loss = loss_all_loss[mask_all > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        ## B *iters, H*W -> B, iters, H*W
        loss_all_loss_mean = (loss_all_loss.view(batch, max_iters, -1) * linear_terms).sum(dim=1).mean()

        # loss_all_loss_mean = loss_all_loss.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_all_loss_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc




def train_progressive_l1mask(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    custom_beta = train_setup.custom_beta

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, _, masks = net(inputs, iters_to_do=max_iters,return_masks=True)
        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            raise NotImplementedError
            outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        internal_mask_loss = net.get_mask_loss(masks)

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean + internal_mask_loss*custom_beta
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    ## DEBUG, screws step.... commit doesnt work
    wandb.log({"mask_loss": internal_mask_loss.item()})

    return train_loss, acc


def train_progressive_flowl1(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    custom_beta = train_setup.custom_beta

    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        outputs_max_iters, iterim_thought, recur_inter = net(inputs, iters_to_do=max_iters,return_recur_inter=True)

        iterim_2,recur_inter = net.recur_block[-1][-1].forward(iterim_thought)

        if alpha != 1:
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            # raise NotImplementedError
            outputs, k,recur_inter_more = get_output_for_more_iters_loss_flow(inputs, iterim_thought, max_iters, net)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
        else:
            recur_inter_more=None
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        # flow_l1_loss = torch.sum(recur_inter.abs())
        flow_l1_loss = torch.sum((iterim_thought - iterim_2).abs())

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean + flow_l1_loss*custom_beta
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    lr_scheduler.step()
    warmup_scheduler.dampen()

    ## DEBUG, screws step.... commit doesnt work
    wandb.log({"flow_loss": flow_l1_loss.item()})
    if recur_inter_more is not None: 
        wandb.log({"flow_loss_more": torch.sum(recur_inter_more.abs()).item()})


    return train_loss, acc



def train_progressive_noise(net, loaders, train_setup, device):
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    k = 0
    problem = train_setup.problem
    clip = train_setup.clip
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    custom_beta = train_setup.custom_beta

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        inputs, targets = inputs.to(device), targets.to(device).long()
        targets = targets.view(targets.size(0), -1)
        if problem == "mazes" or "mask" in problem:
            mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0

        optimizer.zero_grad()

        # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
        # so we save time by settign it equal to 0).
        if alpha != 1:
            outputs_max_iters, _ = net(inputs, iters_to_do=max_iters,alpha_noise=custom_beta)
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)
        else:
            loss_max_iters = torch.zeros_like(targets).float()

        # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
        # so we save time by setting it equal to 0).
        if alpha != 0:
            outputs, k = get_output_for_prog_loss_add_noise(inputs, max_iters, net, custom_beta)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
            loss_progressive = criterion(outputs, targets)
            if alpha == 1:
                outputs_max_iters=outputs
        else:
            loss_progressive = torch.zeros_like(targets).float()

        if problem == "mazes" or "mask" in problem:
            loss_max_iters = (loss_max_iters * mask)
            loss_max_iters = loss_max_iters[mask > 0]
            loss_progressive = (loss_progressive * mask)
            loss_progressive = loss_progressive[mask > 0]

        loss_max_iters_mean = loss_max_iters.mean()
        loss_progressive_mean = loss_progressive.mean()

        loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem)
        correct += torch.amin(predicted == targets, dim=[-1]).sum().item()
        total += targets.size(0)

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total

    # do in training.py
    # lr_scheduler.step()
    # warmup_scheduler.dampen()

    return train_loss, acc