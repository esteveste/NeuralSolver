
import logging
import os
import sys
from collections import OrderedDict

import json

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import deepthinking as dt

from hydra import initialize, compose
from omegaconf import OmegaConf

### rl prepare
import argparse
import os
import random
import time
from distutils.util import strtobool


# from gym.envs.registration import register
from gymnasium.envs.registration import register
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
# import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

import subprocess
## default device to lowest gpu

from eval_utils import *
api = wandb.Api()

device='cuda'

# device = "cpu"


register(
    id="MiniGrid-DoorKey-6x6-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 6},
)
register(
    id="MiniGrid-DoorKey-7x7-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 7},
)

register(
    id="MiniGrid-DoorKey-8x8-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 8},
)

register(
    id="MiniGrid-DoorKey-9x9-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 9},
)
register(
    id="MiniGrid-DoorKey-10x10-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 10},
)

register(
    id="MiniGrid-DoorKey-11x11-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 11},
)
register(
    id="MiniGrid-DoorKey-12x12-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 12},
)

register(
    id="MiniGrid-DoorKey-13x13-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 13},
)
register(
    id="MiniGrid-DoorKey-16x16-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 16},
)
register(
    id="MiniGrid-DoorKey-20x20-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 20},
)

register(
    id="MiniGrid-DoorKey-40x40-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 40},
)

register(
    id="MiniGrid-DoorKey-50x50-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 50},
)

register(
    id="MiniGrid-DoorKey-80x80-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 80},
)

register(
    id="MiniGrid-DoorKey-120x120-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 120},
)
register(
    id="MiniGrid-DoorKey-160x160-v0",
    entry_point="minigrid.envs:DoorKeyEnv",
    kwargs={"size": 160},
)

LIST_SIZES = [7,10,11,13,16,20, 40, 80, 120, 160]

max_minigrid_step =  6400 # 4 * 40**2 dont do more than 40


from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper,ImgObsWrapper
from gymnasium.core import ObservationWrapper, Wrapper

from gym.wrappers import RecordEpisodeStatistics,RecordVideo, TimeLimit
class PytorchWrapper(ObservationWrapper):

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.env.width, self.env.height),  # number of cells
            dtype=float,
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        return obs.transpose(2, 0, 1).astype(float)/10 # 10
    
class PytorchWrapperPOMDP(ObservationWrapper):

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, 7, 7),  # number of cells
            dtype=float,
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        return obs.transpose(2, 0, 1).astype(float)/10 # 10
    

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, )

        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = PytorchWrapper(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        env= TimeLimit(env, max_minigrid_step)

        print("WARNING: No seed set for env!")
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)


        return env

    return thunk

def make_env_pomdb(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, )

        # env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        env = PytorchWrapperPOMDP(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        env= TimeLimit(env, max_minigrid_step)
        
        print("WARNING: No seed set for env!")
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)


        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()


        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        print(envs.single_observation_space)

        n = envs.single_observation_space.shape[1]
        m = envs.single_observation_space.shape[2]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.critic = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # nn.Tanh(),
            layer_init(nn.Linear(self.image_embedding_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            # layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            # nn.Tanh(),
            layer_init(nn.Linear(self.image_embedding_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        x= self.image_conv(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = self.image_conv(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), logits

import pandas as pd 
import wandb
import numpy as np



# group = 'v3_eval_linear_ep20_shufflenetV2_0.5_cifar100'

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name='test_model_config.yaml')
    print(OmegaConf.to_yaml(cfg))

## change current working directory to the directory of the script
os.chdir("./deepthinking/")
os.chdir("./models/")

### needs to be very deep...

## LOAD STUFF


def get_model_path(_run_id,folder_path='../../../outputs/mazes_ablation'):
    model_path = f'{folder_path}/training-{_run_id}/'

    if not os.path.exists(model_path):
        raise Exception(f"path not found! : {model_path}")

    return model_path

def load_model_and_dataloaders(run, model_path, test_size = 13, file_name = 'model_best',test_batch_size=25,smaller_test_batch_size=10,use_compile=False):
    global cfg


    training_args = OmegaConf.load(os.path.join(model_path, ".hydra/config.yaml"))
    cfg_keys_to_load = [("hyp", "alpha"),
                        ("hyp", "epochs"),
                        ("hyp", "lr"),
                        ("hyp", "lr_factor"),
                        ("model", "max_iters"),
                        ("model", "model"),
                        ("hyp", "optimizer"),
                        ("hyp", "train_mode"),
                        ("model", "width")]
    for k1, k2 in cfg_keys_to_load:
        cfg["problem"][k1][k2] = training_args["problem"][k1][k2]
    cfg.problem.train_data = cfg.problem.train_data

    cfg=training_args
    # cfg.problem = training_args.problem
    cfg.problem.model.model_path = model_path
    # 'dt_net_recall_2d_out4'

    # cfg.problem.hyp.test_batch_size=5

    cfg.problem.hyp.test_batch_size=test_batch_size

    if 'lab' in subprocess.getoutput("hostname") and 'prefix' not in cfg.problem.name:
        # we are in RNL, use smaller batch size to avoid memory issues
        cfg.problem.hyp.test_batch_size=smaller_test_batch_size


    cfg.problem.test_data = test_size

    # backwards compatibility to older runs, and for eval....
    cfg.problem.hyp.single_batch = False

    # device = "cuda" if torch.cuda.is_available() else "cpu"


    # assert cfg.wandb_group == run.group, f"group name does not match to cfg file: {run.group} vs {cfg.wandb_group}"


    torch.backends.cudnn.benchmark = True
    if cfg.problem.hyp.save_period is None:
        cfg.problem.hyp.save_period = cfg.problem.hyp.epochs

    cfg.problem.model.model_path = model_path

    ### fix add module. on each key on state dict
    state_dict = torch.load(os.path.join(cfg.problem.model.model_path, f"{file_name}.pth"), map_location=device)
    state_dict['net'] = OrderedDict([(k.replace('module.', '').replace('_orig_mod.', ''), v) for k, v in state_dict['net'].items()])

    torch.save(state_dict, os.path.join(cfg.problem.model.model_path, f"{file_name}2.pth"))



    ####################################################
    # LOAD   Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    cfg.problem.model.model_path = os.path.join(cfg.problem.model.model_path, f"{file_name}2.pth")
    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                    cfg.problem.model,
                                                                                    device, use_data_parallel=False, use_compile=use_compile)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    ####################################################

    return net, loaders


### custom forwards
def test_default2(net, testloader, iters, problem, device, short=True):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs, all_iterin = custom_forward(net, inputs, iters_to_do=max_iters)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem)
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            total += targets.size(0)

            if short:
                break

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc,all_outputs, all_iterin, inputs.cpu(), targets.cpu()

def custom_forward(self, x, iters_to_do, interim_thought=None, **kwargs):
    raise NotImplementedError
    initial_thought = self.projection(x)

    if interim_thought is None:
        interim_thought = initial_thought

    if hasattr(self, "output_size"):
        all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)
        all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        )).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought).view(x.size(0), self.output_size)

            all_outputs[:, i] = out
            all_iterim_thoughts[:, i] = interim_thought
    else:

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        )).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out
            all_iterim_thoughts[:, i] = interim_thought

    # if self.training:
    #     return out, interim_thought

    return all_outputs, all_iterim_thoughts


def test_default_fixed_point(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    fixed_points_sum_final = np.zeros(max_iters)
    fixed_points_max_final = np.zeros(max_iters)

    output_pred_sum_final = np.zeros(max_iters)
    output_pred_max_final = np.zeros(max_iters)

    output_logits_sum_final = np.zeros(max_iters)
    output_logits_max_final = np.zeros(max_iters)

    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs,  fixed_points_sum, fixed_points_max = custom_forward_fixed_point(net, inputs, iters_to_do=max_iters)

            fixed_points_sum_final += fixed_points_sum
            fixed_points_max_final = np.maximum(fixed_points_max_final, fixed_points_max)

            output_pred_sum = np.zeros(max_iters)
            output_pred_max = np.zeros(max_iters)

            output_logits_sum = np.zeros(max_iters)
            output_logits_max = np.zeros(max_iters)

            previous_outputs = all_outputs[:, 0]
            previous_predicted = get_predicted(inputs, all_outputs[:, 0], problem).float()

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem).float()
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

                norm2 = torch.norm(predicted - previous_predicted, dim=list(range(1, len(predicted.shape))), p=2)
                output_pred_sum[i] = torch.sum(norm2).item()
                output_pred_max[i] = torch.max(norm2).item()

                norm2 = torch.norm(outputs - previous_outputs, dim=list(range(1, len(outputs.shape))), p=2)
                output_logits_sum[i] = torch.sum(norm2).item()
                output_logits_max[i] = torch.max(norm2).item()

                previous_outputs = outputs
                previous_predicted = predicted


    
            total += targets.size(0)
            output_pred_sum_final += output_pred_sum
            output_pred_max_final = np.maximum(output_pred_max_final, output_pred_max)

            output_logits_sum_final += output_logits_sum
            output_logits_max_final = np.maximum(output_logits_max_final, output_logits_max)

    fixed_points_sum_final = fixed_points_sum_final / total

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc,all_outputs, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final[1:], output_pred_max_final[1:], output_logits_sum_final[1:], output_logits_max_final[1:]


def test_default_fixed_point_interference(net, testloader, iters, problem, device,iterim_function, interrupt_iter):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    fixed_points_sum_final = np.zeros(max_iters)
    fixed_points_max_final = np.zeros(max_iters)

    output_pred_sum_final = np.zeros(max_iters)
    output_pred_max_final = np.zeros(max_iters)

    output_logits_sum_final = np.zeros(max_iters)
    output_logits_max_final = np.zeros(max_iters)

    assert interrupt_iter < max_iters

    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            all_outputs,  fixed_points_sum, fixed_points_max, last_iterim = custom_forward_fixed_point(net, inputs, iters_to_do=interrupt_iter,return_last_iterim=True)

            interim=iterim_function(last_iterim)

            all_outputs_new, fixed_points_sum_new, fixed_points_max_new = custom_forward_fixed_point(net, inputs, iters_to_do=max_iters-interrupt_iter,interim_thought=interim)

            all_outputs=torch.cat([all_outputs,all_outputs_new],dim=1)
            fixed_points_sum= np.concatenate([fixed_points_sum,fixed_points_sum_new],axis=0)
            fixed_points_max= np.concatenate([fixed_points_max,fixed_points_max_new],axis=0)


            fixed_points_sum_final += fixed_points_sum
            fixed_points_max_final = np.maximum(fixed_points_max_final, fixed_points_max)

            output_pred_sum = np.zeros(max_iters)
            output_pred_max = np.zeros(max_iters)

            output_logits_sum = np.zeros(max_iters)
            output_logits_max = np.zeros(max_iters)

            previous_outputs = all_outputs[:, 0]
            previous_predicted = get_predicted(inputs, all_outputs[:, 0], problem).float()

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                predicted = get_predicted(inputs, outputs, problem).float()
                targets = targets.view(targets.size(0), -1)
                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

                norm2 = torch.norm(predicted - previous_predicted, dim=list(range(1, len(predicted.shape))), p=2)
                output_pred_sum[i] = torch.sum(norm2).item()
                output_pred_max[i] = torch.max(norm2).item()

                norm2 = torch.norm(outputs - previous_outputs, dim=list(range(1, len(outputs.shape))), p=2)
                output_logits_sum[i] = torch.sum(norm2).item()
                output_logits_max[i] = torch.max(norm2).item()

                previous_outputs = outputs
                previous_predicted = predicted


    
            total += targets.size(0)
            output_pred_sum_final += output_pred_sum
            output_pred_max_final = np.maximum(output_pred_max_final, output_pred_max)

            output_logits_sum_final += output_logits_sum
            output_logits_max_final = np.maximum(output_logits_max_final, output_logits_max)

    fixed_points_sum_final = fixed_points_sum_final / total

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc,all_outputs, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final[1:], output_pred_max_final[1:], output_logits_sum_final[1:], output_logits_max_final[1:]


import torch.distributions as tdist

class Discrete(tdist.Bernoulli):
    ## goes through a sigmoid, is more stable gradient

    @property
    def mode(self):
        return torch.round(self.probs)  # >0.5
    
    def rsample(self, sample_shape=torch.Size()):
        # Straight through biased gradient estimator.
        probs = self.probs.expand(sample_shape + self.batch_shape + self.event_shape)
        sample = torch.round(probs)
        sample += (probs - probs.detach())
        return sample
    
    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)


def custom_forward_fixed_point(self, x, iters_to_do, interim_thought=None,return_last_iterim=False, **kwargs):

    if hasattr(self, "projection"):
        initial_thought = self.projection(x)
    else:
        initial_thought = torch.zeros(x.shape[0],self.width, x.shape[2], x.shape[3]).to(x.device)


    # esta linha e' horrivel..............
    if hasattr(self, "output_size") and self.name not in ['NetConvLSTM_LN_Reduce','NetConvLSTM_LN','NetConvLSTM_LN_5L','NetConvLSTM_LN_Reduce_5L','NetConvLSTM_LN_NL']:

        if interim_thought is None:
            interim_thought = initial_thought

        if hasattr(self, "update_layer"):
            raise NotImplementedError

        all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)
        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):

            previous_thought = interim_thought

            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought).view(x.size(0), self.output_size)

            all_outputs[:, i] = out


            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max

            # all_iterim_thoughts[:, i] = interim_thought

    elif hasattr(self, "name") and self.name == "DTNetReduceMessages":
        if interim_thought is None:
            interim_thought = initial_thought
        print("evaluating foward DTNetReduceMessages")
        interim_thought,mask = self.get_iterim_next(interim_thought)

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        all_masks = torch.zeros((x.size(0), iters_to_do, 1, x.size(2), x.size(3))).to(x.device)


        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)


        for i in range(iters_to_do):
            interim_thought_previous = interim_thought
            previous_thought = interim_thought

            
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            # interim_thought = self.recur_block(interim_thought)
            # out = self.head(interim_thought)
            # all_outputs[:, i] = out

            interim_thought_next = self.recur_block(interim_thought)

            # final activation
            interim_thought_next, next_mask = self.get_iterim_next(interim_thought_next)
            
            interim_thought = interim_thought_next * mask + interim_thought_previous * (1-mask)

            out = self.head(interim_thought)
            all_outputs[:, i] = out
            all_masks[:, i] = mask

            mask = next_mask
        

            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max



    elif hasattr(self, "name") and self.name == "DTNetCustom":
        if interim_thought is None:
            interim_thought = initial_thought
        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):
            previous_thought = interim_thought
            if self.recall:
                interim_thought = self.recur_block(torch.cat([interim_thought, x], 1), interim_thought)
            else:
                interim_thought = self.recur_block(interim_thought, interim_thought)

            out = self.head(interim_thought)
            all_outputs[:, i] = out

            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max


    elif hasattr(self, "name") and self.name == "FlowNet":
        if interim_thought is None:
            if self.recall:
                interim_thought = torch.cat([initial_thought, x], 1)
            else:
                interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):
            previous_thought = interim_thought
            interim_thought, recur_inter = self.recur_block(interim_thought) ## fixme if recurrent inter

            out = self.head(interim_thought)
            all_outputs[:, i] = out

            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max



    elif hasattr(self, "name") and (self.name == "DTNetSmallTest" or self.name == "DTNetSmallTest1Conv"):
        interim_thought = torch.zeros(x.shape[0],self.stoch_classes, x.shape[2], x.shape[3]).to(x.device)

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):
            previous_thought = interim_thought

            if self.recall:
                interim_thought_input = torch.cat([interim_thought, x], 1)
            else:
                interim_thought_input = interim_thought
            interim_thought_new = self.recur_block(interim_thought_input)

            if hasattr(self, "update_layer"):
                interim_thought = self.update_layer(interim_thought,interim_thought_new) 
            else:
                interim_thought = interim_thought_new 

            dist = Discrete(logits=interim_thought.permute(0,2,3,1))

            # dist = Discrete2(logits=interim_thought.permute(0,2,3,1))

            # dist = ContinuousBernoulli(logits=interim_thought.permute(0,2,3,1)*2)
            # dist = Bernoulli(logits=interim_thought.permute(0,2,3,1)*2)

            # interim_thought = torch.sigmoid(interim_thought*2)
            # interim_thought = self.adaptive_discrete(interim_thought)


            if self.training:
                interim_thought = dist.sample().permute(0,3,1,2)
            else:
                interim_thought = dist.mode.permute(0,3,1,2)
            


            out = self.head(interim_thought)
            all_outputs[:, i] = out

            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max

            # all_iterim_thoughts[:, i] = interim_thought


    elif hasattr(self, "name") and self.name == "DTNetLSTM":
        if interim_thought is None:
            interim_thought = initial_thought
        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)


        interim_thought_flat = rearrange(interim_thought, 'b c h w -> (b h w) c')
        
        self._input_drop.set_weights(interim_thought_flat)
        self._state_drop.set_weights(interim_thought_flat)
        
        interim_thought_h, c = self.lstm(self._input_drop(interim_thought_flat))
        interim_thought_h = self._state_drop(interim_thought_h)
        interim_thought = rearrange(interim_thought_h, '(b h w) c -> b c h w', b=x.size(0), h=x.size(2), w=x.size(3))

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):
            previous_thought = interim_thought

            if self.recall:
                interim_thought_new = torch.cat([interim_thought, x], 1)
            else:
                interim_thought_new = interim_thought
            interim_thought_new = self.recur_block(interim_thought_new)
            
            interim_thought_new = rearrange(interim_thought_new, 'b c h w -> (b h w) c')
            interim_thought_h, c = self.lstm(self._input_drop(interim_thought_new),(interim_thought_h,c))
            interim_thought_h = self._state_drop(interim_thought_h)
            interim_thought = rearrange(interim_thought_h, '(b h w) c -> b c h w', b=x.size(0), h=x.size(2), w=x.size(3))
            
            out = self.head(interim_thought)
            all_outputs[:, i] = out




            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max

            # all_iterim_thoughts[:, i] = interim_thought


    elif hasattr(self, "name") and self.name == "NetConvLSTM":
        if interim_thought is None:
            interim_thought = initial_thought
        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        mul=2

        for i in range(iters_to_do*mul):
            previous_thought = interim_thought

            if i==0:
                self._input_drop.set_weights(interim_thought)
                self._state_drop.set_weights(interim_thought)

                state = None
            else:
                state = (interim_thought,c)
                

            if self.recall:
                interim_thought_new = torch.cat([self._input_drop(interim_thought), x], 1)
            else:
                assert False, "not implemented"

            interim_thought, c = self.lstm(interim_thought_new,state)
            interim_thought = self._state_drop(interim_thought)

            if i==0:
                state2=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                # should it be state drop from other lstm though?
                state2 = (interim_thought,c2)

            interim_thought, c2 = self.lstm2(self._input_drop(interim_thought),state2)
            interim_thought = self._state_drop(interim_thought)

            if i==0:
                state3=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                state3 = (interim_thought,c3)
            
            interim_thought, c3 = self.lstm3(self._input_drop(interim_thought),state3)
            interim_thought = self._state_drop(interim_thought)

            if i%mul==mul-1:
                out = self.head(interim_thought)
                all_outputs[:, i//mul] = out

                # frobenious norm
                norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
                norm2_sum = torch.sum(norm2).item()
                norm2_max = torch.max(norm2).item()

                fixed_points_sum[i//mul] = norm2_sum
                fixed_points_max[i//mul] = norm2_max


    elif hasattr(self, "name") and self.name in ["NetConvLSTM_LN","NetConvLSTM_LN_Reduce"]:
        if interim_thought is None:
            interim_thought = initial_thought
        
        if self.name == "NetConvLSTM_LN_Reduce" and hasattr(self, "output_size"):
            all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)
        else:
            if len(x.shape)==4:
                if hasattr(self, "flatten") and self.flatten:
                    all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size*x.size(2)*x.size(3))).to(x.device)
                else:
                    all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2), x.size(3))).to(x.device)
            elif len(x.shape)==3:
                all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2))).to(x.device)
            else:
                assert False, "not implemented"

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        mul=5 # para ser equivalente a 3 lstms
        self.lstm.sample_mask(interim_thought.device)

        lstm_inp1 = self.lstm.forward_input(x)

        for i in range(iters_to_do*mul):
            if i%mul==0:
                previous_thought = interim_thought

            if i==0:
                self._state_drop.set_weights(interim_thought)
                state = None
            else:
                state = (interim_thought,c)
                
                
            interim_thought, c = self.lstm(lstm_inp1,state)
            interim_thought = self._state_drop(interim_thought)


            if i%mul==mul-1:
                out = self.head(interim_thought)

                if  self.name == "NetConvLSTM_LN_Reduce" and hasattr(self, "output_size"): # expects reduce...
                    out = out.view(x.size(0), self.output_size)

                if hasattr(self, "flatten") and self.flatten:
                    out = out.flatten(1)

                all_outputs[:, i//mul] = out

                # frobenious norm
                norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
                norm2_sum = torch.sum(norm2).item()
                norm2_max = torch.max(norm2).item()

                fixed_points_sum[i//mul] = norm2_sum
                fixed_points_max[i//mul] = norm2_max


    elif hasattr(self, "name") and self.name in ["NetConvLSTM_LN_NL"]:
        if interim_thought is None:
            interim_thought = initial_thought
        
        if self.name == "NetConvLSTM_LN_NL_REDUCE" and hasattr(self, "output_size"):
            all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size)).to(x.device)
        else:
            if len(x.shape)==4:
                if hasattr(self, "flatten") and self.flatten:
                    all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size*x.size(2)*x.size(3))).to(x.device)
                else:
                    all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2), x.size(3))).to(x.device)
            elif len(x.shape)==3:
                all_outputs = torch.zeros((x.size(0), iters_to_do, self.output_size, x.size(2))).to(x.device)
            else:
                assert False, "not implemented"

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        mul=5 # para ser equivalente a 3 lstms
        lstm_inputs = []
        for lstm in self.lstm_list:
            lstm.sample_mask(interim_thought.device)
            lstm_inputs.append(lstm.forward_input(x))

        c_states = [None for i in range(self.nr_lstm_layers)]
        lstm_counter = 0

        for i in range(iters_to_do*mul):
            if i%mul==0:
                previous_thought = interim_thought

            if i==0:
                self._state_drop.set_weights(interim_thought)



            if c_states[lstm_counter] is None:
                state = None
            else:
                state = (interim_thought,c_states[lstm_counter])
                

            # if self.recall:
            #     interim_thought_new = torch.cat([self._input_drop(interim_thought), x], 1)
            # else:
            #     assert False, "not implemented"

            interim_thought_new, c = self.lstm_list[lstm_counter](lstm_inputs[lstm_counter],state)
            interim_thought_new = self._state_drop(interim_thought_new)

            if self.use_skip and i%self.skip_mul==self.skip_mul-1:
                interim_thought = interim_thought + interim_thought_new
            else:
                interim_thought = interim_thought_new

            c_states[lstm_counter] = c

            lstm_counter += 1
            lstm_counter = lstm_counter % self.nr_lstm_layers

            if i%mul==mul-1:
                out = self.head(interim_thought)

                if  self.name == "NetConvLSTM_LN_Reduce" and hasattr(self, "output_size"): # expects reduce...
                    out = out.view(x.size(0), self.output_size)

                if hasattr(self, "flatten") and self.flatten:
                    out = out.flatten(1)

                all_outputs[:, i//mul] = out

                # frobenious norm
                norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
                norm2_sum = torch.sum(norm2).item()
                norm2_max = torch.max(norm2).item()

                fixed_points_sum[i//mul] = norm2_sum
                fixed_points_max[i//mul] = norm2_max



    elif hasattr(self, "name") and self.name == "NetConvLSTM_LN_5L":
        if interim_thought is None:
            interim_thought = initial_thought
        
        
        if len(x.shape)==4:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        elif len(x.shape)==3:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)
        else:
            assert False, "not implemented"

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        mul=1
        self.lstm.sample_mask(interim_thought.device)
        self.lstm2.sample_mask(interim_thought.device)
        self.lstm3.sample_mask(interim_thought.device)
        self.lstm4.sample_mask(interim_thought.device)
        self.lstm5.sample_mask(interim_thought.device)

        lstm_inp1 = self.lstm.forward_input(x)
        lstm_inp2 = self.lstm2.forward_input(x)
        lstm_inp3 = self.lstm3.forward_input(x)
        lstm_inp4 = self.lstm4.forward_input(x)
        lstm_inp5 = self.lstm5.forward_input(x)

        for i in range(iters_to_do*mul):
            if i%mul==0:
                previous_thought = interim_thought

            if i==0:
                self._state_drop.set_weights(interim_thought)
                state = None
            else:
                state = (interim_thought,c)
      
            interim_thought, c = self.lstm(lstm_inp1,state)
            interim_thought = self._state_drop(interim_thought)


            if i==0:
                state2=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                # should it be state drop from other lstm though?
                state2 = (interim_thought,c2)

            interim_thought, c2 = self.lstm2(lstm_inp2,state2)
            interim_thought = self._state_drop(interim_thought)


            if i==0:
                state3=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                state3 = (interim_thought,c3)
            
            interim_thought, c3 = self.lstm3(lstm_inp3,state3)
            interim_thought = self._state_drop(interim_thought)


            if i==0:
                state4=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                state4 = (interim_thought,c4)
            
            interim_thought, c4 = self.lstm4(lstm_inp4,state4)
            interim_thought = self._state_drop(interim_thought)

            if i==0:
                state5=(interim_thought,torch.zeros_like(c).to(c.device))
            else:
                state5 = (interim_thought,c5)
            
            interim_thought, c5 = self.lstm5(lstm_inp5,state5)
            interim_thought = self._state_drop(interim_thought)
            
            if i%mul==mul-1:
                out = self.head(interim_thought)
                all_outputs[:, i//mul] = out

                # frobenious norm
                norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
                norm2_sum = torch.sum(norm2).item()
                norm2_max = torch.max(norm2).item()

                fixed_points_sum[i//mul] = norm2_sum
                fixed_points_max[i//mul] = norm2_max

    elif hasattr(self, "name"):
        ### and name not in previous ifs
        raise NotImplementedError

    else:
        if interim_thought is None:
            interim_thought = initial_thought

        
        if len(x.shape)==4:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        elif len(x.shape)==3:
            all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)
        else:
            assert False, "not implemented"


        # all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)
        # all_iterim_thoughts = torch.zeros((x.size(0), iters_to_do, interim_thought.size(1), interim_thought.size(2), interim_thought.size(3)
        # )).to(x.device)

        fixed_points_sum = np.zeros(iters_to_do)
        fixed_points_max = np.zeros(iters_to_do)

        for i in range(iters_to_do):
            previous_thought = interim_thought

            if self.recall:
                interim_thought_input = torch.cat([interim_thought, x], 1)
            else:
                interim_thought_input = interim_thought
            interim_thought_new = self.recur_block(interim_thought_input)

            if hasattr(self, "update_layer"):
                interim_thought = self.update_layer(interim_thought,interim_thought_new) 
            else:
                interim_thought = interim_thought_new 


            out = self.head(interim_thought)
            all_outputs[:, i] = out

            # frobenious norm
            norm2 = torch.norm(interim_thought - previous_thought, dim=list(range(1, len(interim_thought.shape))), p=2)
            norm2_sum = torch.sum(norm2).item()
            norm2_max = torch.max(norm2).item()

            fixed_points_sum[i] = norm2_sum
            fixed_points_max[i] = norm2_max

            # all_iterim_thoughts[:, i] = interim_thought


    # if self.training:
    #     return out, interim_thought
    if return_last_iterim:
        return all_outputs, fixed_points_sum, fixed_points_max, interim_thought
    else:
        return all_outputs, fixed_points_sum, fixed_points_max


# ### start here FIXME repeated
# def get_predicted(inputs, outputs, problem):
#     outputs = outputs.clone()
#     predicted = outputs.argmax(1)
#     predicted = predicted.view(predicted.size(0), -1)
#     if problem == "mazes" or "mask" in problem:
        
#         ## fazemos esta treta nos mazes, why?
#         ## isto forca a que o output seja examente dentro das linhas do input da maze
#         ## mas precisamos mesmo disto?
#         predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))


#     elif problem == "chess":
#         outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
#         top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
#         top_2 = einops.repeat(top_2, "n -> n k", k=8)
#         top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
#         outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
#         outputs[:, 0] = -float("Inf")
#         predicted = outputs.argmax(1)

#     return predicted

# def get_probs(inputs, outputs, problem,temperature=1):
#     outputs = outputs.clone()
#     # predicted = outputs.argmax(1)
#     predicted = torch.softmax(outputs/temperature, dim=1)
#     ## get last class (1)
#     predicted = predicted[:,1]
#     # predicted = predicted.view(predicted.size(0), -1)
#     if problem == "mazes" or "mask" in problem:
        
#         ## fazemos esta treta nos mazes, why?
#         ## isto forca a que o output seja examente dentro das linhas do input da maze
#         ## mas precisamos mesmo disto?
#         predicted = predicted * (inputs.max(1)[0])  #.view(inputs.size(0), -1))


#     # elif problem == "chess":
#         # outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
#         # top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
#         # top_2 = einops.repeat(top_2, "n -> n k", k=8)
#         # top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
#         # outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
#         # outputs[:, 0] = -float("Inf")
#         # predicted = outputs.argmax(1)

#     return predicted

## long term evaluation

from deepthinking.utils import testing
from deepthinking.utils.testing import * 

def supervised_test(net,loaders, max_iters, test_fn=testing.test_default):
    test_iterations = list(range(1,max_iters+1))
    # out = testing.test_max_conf(net, loaders["test"], test_iterations, cfg.problem.name, device)
    out = test_fn(net, loaders["test"], test_iterations, cfg.problem.name, device)

    l = []
    for i in range(1,max_iters+1):
        l.append(out[i])

    return l



### not really used anymore
def test_default_nomemory(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            corrects_batch = net.forward_nomemory(inputs, targets,lambda x,y:get_predicted(x, y, problem), iters_to_do=max_iters)

            # for i in range(all_outputs.size(1)):
            #     outputs = all_outputs[:, i]
            #     predicted = get_predicted(inputs, outputs, problem)
            #     targets = targets.view(targets.size(0), -1)
            #     corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()

            corrects += corrects_batch

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc


def supervised_fixed_point(net,loaders, max_iters):
    test_iterations = list(range(1,max_iters+1))
    # out = testing.test_max_conf(net, loaders["test"], test_iterations, cfg.problem.name, device)
    out,_, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final = test_default_fixed_point(net, loaders["test"], test_iterations, cfg.problem.name, device)



    l = []
    for i in range(1,max_iters+1):
        l.append(out[i])

    return l, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final



def supervised_fixed_point_interference(net,loaders, max_iters, interference_iter, intereference_function):
    test_iterations = list(range(1,max_iters+1))


    # out = testing.test_max_conf(net, loaders["test"], test_iterations, cfg.problem.name, device)
    out,_, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final = test_default_fixed_point_interference(net, loaders["test"], test_iterations, cfg.problem.name, device,
                                                                                                                                                                                                  intereference_function, interference_iter, )

    l = []
    for i in range(1,max_iters+1):
        l.append(out[i])

    return l, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final


### out is the direct output score on each iteration(without max confidence)

### rl run eval, make envs

def rl_make_envs(size,num_envs=16,pomdp=False):
    env_id=f"MiniGrid-DoorKey-{size}x{size}-v0"
    # env_id="MiniGrid-DoorKey-16x16-v0"
    # env_id="MiniGrid-DoorKey-8x8-v0"
    # env_id="MiniGrid-DoorKey-10x10-v0"
    seed=0
    capture_video=False
    run_name="test"

    if pomdp:
        make_class = make_env_pomdb
    else:
        make_class = make_env

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_class(env_id, seed + i, i, capture_video, run_name) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    return envs

def rl_doorkey_eval(net,envs,greedy=False):
    num_envs=len(envs.envs)

    obs__,_=envs.reset()
    next_obs = torch.Tensor(obs__).to(device)

    net.eval()


    from tqdm import tqdm

    # nr_steps = 0
    eval_episodes = 100
    episodes_done = 0

    done = [False] * num_envs

    total_episode_rewards = []
    total_episode_lengths = []
    sum_rewards=np.zeros((num_envs,))
    ep_len =  np.zeros((num_envs,))

    max_iters = 30

    net.eval()

    # first=True

    # follow_1_env = torch.zeros(max_steps,size,size,3)

    with tqdm(total=eval_episodes) as pbar:
    # for nr_step in tqdm(range(max_steps)):
        while episodes_done < eval_episodes:

                
            with torch.no_grad():
                # follow_1_env[nr_step] = next_obs[0].permute(1,2,0).cpu()
                all_outputs = net(next_obs, iters_to_do=max_iters)


                probs = Categorical(logits=all_outputs[:,-1].cpu())
                # if action is None:
                action = probs.sample()
                greedy_action = torch.argmax(all_outputs[:,-1].cpu(),dim=-1)

                if greedy:
                    action = greedy_action

                ## amazing how sample works much better than greedy action!



            action = torch.clamp(action,0,5) # clip invalid actions

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

            # print(info)

            done = [t or d for t, d in zip(terminated, truncated)]
            # done = done[0]

            next_obs = torch.Tensor(next_obs).to(device)

            # print(reward)
            sum_rewards += reward
            ep_len+=1

            for i, d in enumerate(done):
                if d:
                    total_episode_rewards.append(sum_rewards[i])
                    total_episode_lengths.append(ep_len[i])

                    sum_rewards[i]=0
                    ep_len[i]=0

                    episodes_done+=1
                    pbar.update(1)


    # np.mean(total_episode_rewards), np.std(total_episode_rewards), len(total_episode_rewards[1:])
    return total_episode_rewards, total_episode_lengths
    ## eval of this is weirder and harder since is parallel, and slightly unfair if we do not count the cut off episodes

def get_agent(envs, run,pomdp=True):
    if pomdp:
        model_path = f'rl_dist_ppo_pomdp_{run.name}.pt'
    else:
        model_path = f'rl_dist_ppo_{run.name}.pt'

    file_path = f"../../../outputs/{model_path}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file {file_path} not found.")

    agent= Agent(envs).to(device)
    agent.load_state_dict(torch.load(file_path, map_location=device))

    return agent


### rl run eval, make envs
def rl_doorkey_agent(agent,envs,greedy=False):
    num_envs=len(envs.envs)


    obs__,_=envs.reset()
    next_obs = torch.Tensor(obs__).to(device)



    from tqdm import tqdm

    # nr_steps = 0
    eval_episodes = 100
    episodes_done = 0

    done = [False] * num_envs

    total_episode_rewards = []
    total_episode_lengths = []
    sum_rewards=np.zeros((num_envs,))
    ep_len =  np.zeros((num_envs,))

    max_iters = 30



    # first=True

    # follow_1_env = torch.zeros(max_steps,size,size,3)

    with tqdm(total=eval_episodes) as pbar:
    # for nr_step in tqdm(range(max_steps)):
        while episodes_done < eval_episodes:

                
            with torch.no_grad():
                # follow_1_env[nr_step] = next_obs[0].permute(1,2,0).cpu()
                # all_outputs = net(next_obs, iters_to_do=max_iters)
                action, _, _, _, logits = agent.get_action_and_value(next_obs)
                greedy_action = torch.argmax(logits, dim=1)

                if greedy:
                    action = greedy_action

                ## amazing how sample works much better than greedy action!



            action = torch.clamp(action,0,5) # clip invalid actions

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())

            # print(info)

            done = [t or d for t, d in zip(terminated, truncated)]
            # done = done[0]

            next_obs = torch.Tensor(next_obs).to(device)

            # print(reward)
            sum_rewards += reward
            ep_len+=1

            for i, d in enumerate(done):
                if d:
                    total_episode_rewards.append(sum_rewards[i])
                    total_episode_lengths.append(ep_len[i])

                    sum_rewards[i]=0
                    ep_len[i]=0

                    episodes_done+=1
                    pbar.update(1)


    # np.mean(total_episode_rewards), np.std(total_episode_rewards), len(total_episode_rewards[1:])
    return total_episode_rewards, total_episode_lengths
    ## eval of this is weirder and harder since is parallel, and slightly unfair if we do not count the cut off episodes


def filter_runs_exist_path(runs,folder_path='../../../outputs/mazes_ablation'):
    fil_runs = []
    for run in runs:
        try:

            
            ## no idea why
            if '_content' in run.config:
                run_config = run.config['_content']
            else:
                run_config = run.config

            get_model_path(run_config['run_id'],folder_path=folder_path)
            fil_runs.append(run)
        except:
            pass
    return fil_runs


## define evaluation

def eval_supervised(run,run_id,size=13,max_iters=100,plot_name='evals/{}_size_test',file_name='model_best',folder_path='../../../outputs/mazes_ablation',test_batch_size=25,use_compile=False):
    start_time = time.time()
    ## no idea why
    # if '_content' in run.config:
    #     run_config = run.config['_content']
    # else:
    #     run_config = run.config

    # run_id = run_config['run_id']


    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size,file_name=file_name,test_batch_size=test_batch_size,use_compile=use_compile)

    out = supervised_test(net,loaders,max_iters=max_iters)
    
    x_values, y_values = np.arange(len(out)), out
    plot_name = plot_name.format(size)

    # run.update()

    idx = np.argmax(y_values)

    run.summary[plot_name+'_max'] = y_values[idx]
    run.summary[plot_name+'_max_iter'] = x_values[idx]

    run.summary[plot_name+'_final'] = y_values[-1]
    run.summary[plot_name+'_time'] = time.time()-start_time

    wandb_save_plot(run, plot_name,x_values, y_values)


## refactor this
### FIXME not really used
def eval_supervised_nomemory(run,size=13,max_iters=100,plot_name='evals/{}_size_test',file_name='model_best',folder_path='../../../outputs/mazes_ablation',test_batch_size=25,use_compile=False):
    start_time = time.time()
    ## no idea why
    if '_content' in run.config:
        run_config = run.config['_content']
    else:
        run_config = run.config

    run_id = run_config['run_id']


    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size,file_name=file_name,test_batch_size=test_batch_size,use_compile=use_compile)

    out = supervised_test(net,loaders,max_iters=max_iters,test_fn=test_default_nomemory)
    
    x_values, y_values = np.arange(len(out)), out
    plot_name = plot_name.format(size)

    # run.update()

    idx = np.argmax(y_values)

    run.summary[plot_name+'_max'] = y_values[idx]
    run.summary[plot_name+'_max_iter'] = x_values[idx]

    run.summary[plot_name+'_final'] = y_values[-1]
    run.summary[plot_name+'_time'] = time.time()-start_time

    wandb_save_plot(run, plot_name,x_values, y_values)




def eval_supervised_fixed_point(run,size=13,max_iters=100,file_name='model_best',folder_path='../../../outputs/mazes_ablation'):
    run_id = run.config['_content']['run_id']

    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size, file_name=file_name)

    out = supervised_fixed_point(net,loaders,max_iters=max_iters)

    out, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final = out

    
    x_values = np.arange(len(fixed_points_sum_final))
    x_out_values = np.arange(len(output_logits_max_final))


    wandb_save_plot(run, f'evals/size_{size}_fixed_point_mean',x_values, fixed_points_sum_final)
    wandb_save_plot(run, f'evals/size_{size}_fixed_point_max',x_values, fixed_points_max_final)
    wandb_save_plot(run, f'evals/size_{size}_fixed_point_pred_mean',x_out_values, output_pred_sum_final)
    wandb_save_plot(run, f'evals/size_{size}_fixed_point_pred_max',x_out_values, output_pred_max_final)
    wandb_save_plot(run, f'evals/size_{size}_fixed_point_logits_mean',x_out_values, output_logits_sum_final)
    wandb_save_plot(run, f'evals/size_{size}_fixed_point_logits_max',x_out_values, output_logits_max_final)


def eval_supervised_and_fixed_point(run,size=13,max_iters=100,plot_name='evals/{}_size_test',file_name='model_best',folder_path='../../../outputs/mazes_ablation',test_batch_size=25):
    
    ## no idea why
    if '_content' in run.config:
        run_config = run.config['_content']
    else:
        run_config = run.config

    run_id = run_config['run_id']

    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size,file_name=file_name,test_batch_size=test_batch_size)

    # out = supervised_test(net,loaders,max_iters=max_iters)
    out = supervised_fixed_point(net,loaders,max_iters=max_iters)
    out, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final = out

    ### supervised plots
    x_values, y_values = np.arange(len(out)), out
    plot_name = plot_name.format(size)

    # run.update()

    idx = np.argmax(y_values)

    run.summary[plot_name+'_max'] = y_values[idx]
    run.summary[plot_name+'_max_iter'] = x_values[idx]

    run.summary[plot_name+'_final'] = y_values[-1]


    wandb_save_plot(run, plot_name,x_values, y_values)

    # ### fixed point, FIXME
    # x_values = np.arange(len(fixed_points_sum_final))
    # x_out_values = np.arange(len(output_logits_max_final))


    # wandb_save_plot(run, plot_name+'fixed_point_mean',x_values, fixed_points_sum_final)
    # wandb_save_plot(run, plot_name+'fixed_point_max',x_values, fixed_points_max_final)
    # wandb_save_plot(run, plot_name+'fixed_point_pred_mean',x_out_values, output_pred_sum_final)
    # wandb_save_plot(run, plot_name+'fixed_point_pred_max',x_out_values, output_pred_max_final)
    # wandb_save_plot(run, plot_name+'fixed_point_logits_mean',x_out_values, output_logits_sum_final)
    # wandb_save_plot(run, plot_name+'fixed_point_logits_max',x_out_values, output_logits_max_final)
    

# supervised_fixed_point_interference

def eval_supervised_and_fixed_point_gaussian_interference(run,interference_step,std_range=[0.0001,0.001,0.01,0.03,0.06,0.1,0.3,0.6,1,3,6,10],
                                    size=13,max_iters=100,plot_name_template='evals/interf_{}_size_{}_std',file_name='model_best',folder_path='../../../outputs/mazes_ablation'):
    
    """
    plot_name in std is the individual value or all

    example interf_13_size_all_std
    or interf_13_size_0.01_std
    """

    run_id = run.config['_content']['run_id']
    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size,file_name=file_name)

    max_values_list = []
    final_values_list = []

    final_fixed_point_list = []

    for std in tqdm(std_range,desc='std'):
        interference_function = lambda x: x + torch.empty_like(x).normal_(mean=0,std=std) # executed only in same block so is okay

        # out = supervised_test(net,loaders,max_iters=max_iters)
        # out = supervised_fixed_point(net,loaders,max_iters=max_iters)
        out = supervised_fixed_point_interference(net,loaders,max_iters=max_iters,interference_iter=interference_step,intereference_function=interference_function)
        out, fixed_points_sum_final, fixed_points_max_final, output_pred_sum_final, output_pred_max_final, output_logits_sum_final, output_logits_max_final = out



        ### supervised plots
        x_values, y_values = np.arange(len(out)), out
        plot_name = plot_name_template.format(size,std)

        # run.update()

        idx = np.argmax(y_values)

        run.summary[plot_name+'_max'] = y_values[idx]
        run.summary[plot_name+'_max_iter'] = x_values[idx]

        run.summary[plot_name+'_final'] = y_values[-1]


        wandb_save_plot(run, plot_name,x_values, y_values)

        max_values_list.append(y_values[idx])
        final_values_list.append(y_values[-1])

        ### fixed point
        x_values = np.arange(len(fixed_points_sum_final))
        # x_out_values = np.arange(len(output_logits_max_final))

        final_fixed_point_list.append(fixed_points_max_final[-1])

        wandb_save_plot(run, plot_name+'_fixed_point_mean',x_values, fixed_points_sum_final)
        # wandb_save_plot(run, plot_name+'fixed_point_max',x_values, fixed_points_max_final)
        # wandb_save_plot(run, plot_name+'fixed_point_pred_mean',x_out_values, output_pred_sum_final)
        # wandb_save_plot(run, plot_name+'fixed_point_pred_max',x_out_values, output_pred_max_final)
        # wandb_save_plot(run, plot_name+'fixed_point_logits_mean',x_out_values, output_logits_sum_final)
        # wandb_save_plot(run, plot_name+'fixed_point_logits_max',x_out_values, output_logits_max_final)
    
    plot_name = plot_name_template.format(size,"all")
    wandb_save_plot(run, plot_name+"_max_values",std_range, max_values_list)
    wandb_save_plot(run, plot_name+"_final_values",std_range, final_values_list)
    wandb_save_plot(run, plot_name+'_fixed_point_mean',std_range, final_fixed_point_list)

# def eval_supervised_59(run):
#     run_id = run.config['_content']['run_id']

#     net,loaders = load_model_and_dataloaders(run, get_model_path(run_id),test_size=59)

#     out = supervised_test(net,loaders,max_iters=1000)

    
#     wandb_save_plot(run, 'evals/59x59_maze_test', np.arange(len(out)), out)
    

## doorkey
def eval_agent_doorkey(run,size=7,greedy=False):

    plus_string=''
    if greedy:
        plus_string='_greedy'

    envs = rl_make_envs(size,16,pomdp=False)
    agent = get_agent(envs,run, pomdp=False)
    total_episode_rewards, total_episode_lengths = rl_doorkey_agent(agent,envs,greedy=greedy)
    run.summary[f'evals/doorkey_mean_reward{plus_string}'] = np.mean(total_episode_rewards)
    run.summary[f'evals/doorkey_std_reward{plus_string}'] = np.std(total_episode_rewards)
    run.summary[f'evals/doorkey_mean_length{plus_string}'] = np.mean(total_episode_lengths)
    run.summary[f'evals/doorkey_std_length{plus_string}'] = np.std(total_episode_lengths)


def eval_agent_doorkey_fixedsize(run,greedy=False):
    size=int(run.config['env_id'].split('-')[2].split('x')[0])
    eval_agent_doorkey(run,size=size,greedy=greedy)

def eval_agent_pomdp_doorkey_plot(run,greedy=False):
    x_values = []
    y_values = []
    lenght_values = []

    plus_string=''
    if greedy:
        plus_string='_greedy'


    for size in LIST_SIZES:
        envs = rl_make_envs(size,16,pomdp=True)
        agent = get_agent(envs,run)
        total_episode_rewards, total_episode_lengths = rl_doorkey_agent(agent,envs,greedy=greedy)
        run.summary[f'evals/doorkey_{size}x{size}_mean_reward{plus_string}'] = np.mean(total_episode_rewards)
        run.summary[f'evals/doorkey_{size}x{size}_std_reward{plus_string}'] = np.std(total_episode_rewards)
        run.summary[f'evals/doorkey_{size}x{size}_mean_length{plus_string}'] = np.mean(total_episode_lengths)
        run.summary[f'evals/doorkey_{size}x{size}_std_length{plus_string}'] = np.std(total_episode_lengths)
        x_values.append(size)
        y_values.append(np.mean(total_episode_rewards))
        lenght_values.append(np.mean(total_episode_lengths))
    
    wandb_save_plot(run, 'evals/doorkey_pomdp_mean_reward'+plus_string, x_values, y_values)
    wandb_save_plot(run, 'evals/doorkey_pomdp_mean_length'+plus_string, x_values, lenght_values)




# ----------

# ### start here FIXME repeated
# def get_predicted(inputs, outputs, problem):
#     outputs = outputs.clone()
#     predicted = outputs.argmax(1)
#     predicted = predicted.view(predicted.size(0), -1)
#     if problem == "mazes" or "mask" in problem:
        
#         ## fazemos esta treta nos mazes, why?
#         ## isto forca a que o output seja examente dentro das linhas do input da maze
#         ## mas precisamos mesmo disto?
#         predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))


#     elif problem == "chess":
#         outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
#         top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
#         top_2 = einops.repeat(top_2, "n -> n k", k=8)
#         top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
#         outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
#         outputs[:, 0] = -float("Inf")
#         predicted = outputs.argmax(1)

#     return predicted

# def get_probs(inputs, outputs, problem,temperature=1):
#     outputs = outputs.clone()
#     # predicted = outputs.argmax(1)
#     predicted = torch.softmax(outputs/temperature, dim=1)
#     ## get last class (1)
#     predicted = predicted[:,1]
#     # predicted = predicted.view(predicted.size(0), -1)
#     if problem == "mazes" or "mask" in problem:
        
#         ## fazemos esta treta nos mazes, why?
#         ## isto forca a que o output seja examente dentro das linhas do input da maze
#         ## mas precisamos mesmo disto?
#         predicted = predicted * (inputs.max(1)[0])  #.view(inputs.size(0), -1))


#     # elif problem == "chess":
#         # outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
#         # top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
#         # top_2 = einops.repeat(top_2, "n -> n k", k=8)
#         # top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
#         # outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
#         # outputs[:, 0] = -float("Inf")
#         # predicted = outputs.argmax(1)

#     return predicted


# %matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# def make_animation(B=1,frames=250,interval=5):

#     fig, ax = plt.subplots()
    
#     # create numpy array for the image for the given frame
#     images = [get_probs(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu().numpy() for frame in range(frames)]
    
#     def animate(frame):
#         # get the image for the given frame
#         image = images[frame]

#         # clear previous image and plot the new image
#         ax.clear()
#         ax.imshow(image)
#         ax.set_title(f"frame {frame}")

#         # return the new image
#         return ax
#     return FuncAnimation(fig, animate, frames=frames, interval=interval)


## make enum class with colors
from enum import Enum
class ANIMATION_TYPES(Enum):
    DIFF_INTER_1_and_LAST = 1
    DIFF_INTER = 2
    PREDICTED = 3
    DIFF_PREDICTED = 4
    DIFF_INTER_1_and_LAST_2 = 5
    PREDICTED_FLAT = 6


def get_transformed_images(net_values, type,B=1,frames=200):

    out_acc,all_outs,all_iterin, _input, _output = net_values

    
    if type == ANIMATION_TYPES.DIFF_INTER_1_and_LAST:
        images = [torch.abs(all_iterin[B,frame].cpu()
                - 
                all_iterin[B,-1].cpu()).sum(0).numpy() for frame in range(frames)]

    elif type == ANIMATION_TYPES.DIFF_INTER:
        images = [torch.abs(all_iterin[B,frame].cpu()
                - 
                all_iterin[B,frame+1].cpu()).sum(0).numpy() for frame in range(frames)]

    elif type == ANIMATION_TYPES.DIFF_INTER_1_and_LAST_2:
        images = [
                (torch.minimum(
                    torch.abs(all_iterin[B,-1].cpu() - all_iterin[B,frame].cpu()),
                    torch.abs(all_iterin[B,-2].cpu() - all_iterin[B,frame].cpu())))
                .sum(0).numpy() for frame in range(frames)]

    elif type == ANIMATION_TYPES.PREDICTED:
        images = [(get_predicted(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu())
            .reshape(int(all_outs.shape[-1]),int(all_outs.shape[-1])) ## due to original function view..
            .numpy() for frame in range(frames)]
        
    elif type == ANIMATION_TYPES.PREDICTED_FLAT:
        images = [(get_predicted(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu())
            # .reshape(int(all_outs.shape[-1]),int(all_outs.shape[-1])) ## due to original function view..
            .numpy() for frame in range(frames)]
    
    elif type == ANIMATION_TYPES.DIFF_PREDICTED:
        images = [torch.abs(get_predicted(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu()
            - 
            get_predicted(_input.cpu(), all_outs[:,frame+1].cpu(), cfg.problem.name)[B].cpu())
            .reshape(int(all_outs.shape[-1]),int(all_outs.shape[-1])) ## due to original function view..
            .numpy() for frame in range(frames)]

    else:
        raise Exception("type not implemented")
    
    return images
    

def _make_animation(net_values, type,B=1,frames=200,interval=5,normalize=True):

    images = get_transformed_images(net_values, type,B=B,frames=frames)

    # create numpy array for the image for the given frame
    # images = [torch.abs(get_probs(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu()
    #             - 
    #             get_probs(_input.cpu(), all_outs[:,frame+1].cpu(), cfg.problem.name)[B].cpu()).numpy() for frame in range(frames)]


    # images = [torch.abs(get_predicted(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu()
    #             - 
    #             get_predicted(_input.cpu(), all_outs[:,frame+1].cpu(), cfg.problem.name)[B].cpu())
    #             .reshape(int(all_outs.shape[-1]),int(all_outs.shape[-1])) ## due to original function view..
    #             .numpy() for frame in range(frames)]
    

    # images = [torch.abs(get_probs(_input.cpu(), all_outs[:,frame].cpu(), cfg.problem.name)[B].cpu()
    #         - 
    #         get_probs(_input.cpu(), all_outs[:,-1].cpu(), cfg.problem.name)[B].cpu()).numpy() for frame in range(frames)]


    ### internal features
    # images = [torch.abs(all_iterin[B,frame].cpu()
    #         - 
    #         all_iterin[B,frame+1].cpu()).sum(0).numpy() for frame in range(frames)]
    ### L2 doest solve anything

    # ## difference between initial and final

    # images = [torch.abs(all_iterin[B,frame].cpu()
    #         - 
    #         all_iterin[B,-1].cpu()).sum(0).numpy() for frame in range(frames)]

    # images = [torch.abs(all_iterin[B,-1].cpu()
    #         - 
    #         all_iterin[B,frame].cpu()).sum(0).numpy() for frame in range(frames)]


    ### para tentar evitar os ciclos, ver diferenca entre -1 e -2, escolher minimo
    # images = [
    #         (torch.minimum(
    #             torch.abs(all_iterin[B,-1].cpu() - all_iterin[B,frame].cpu()),
    #             torch.abs(all_iterin[B,-2].cpu() - all_iterin[B,frame].cpu())))
    #         .sum(0).numpy() for frame in range(frames)]

    # images = [
    #         (torch.minimum(
    #             torch.abs(all_iterin[B,frame].cpu() - all_iterin[B,frame+1].cpu()),
    #             torch.abs(all_iterin[B,frame].cpu() - all_iterin[B,frame+2].cpu())))
    #         .sum(0).numpy() for frame in range(frames)]


    return make_animation_from_images(images,frames=frames,interval=interval,normalize=normalize)

def make_animation_from_images(images, frames=200,interval=5,normalize=True):
    fig, ax = plt.subplots()

    ## normalize images
    # find max
    max_val = max([np.max(image) for image in images])
    min_val = min([np.min(image) for image in images])
    # # normalize
    # images = [(image-min_val)/(max_val-min_val) for image in images]

    print("anim shape",images[0].shape)

    def animate(frame):
        # get the image for the given frame
        image = images[frame]

        # clear previous image and plot the new image
        ax.clear()
        
        plt.axis('off')
        ax.set_title(f"iteration step: {frame}")

        # # normalize
        if normalize:
            ax.imshow(image)
        else:
            ax.imshow(image,vmin=min_val,vmax=max_val)
        


        # return the new image
        return ax
    return FuncAnimation(fig, animate, frames=frames, interval=interval)

# from IPython.display import Image
# fname='animation_diff_int_imitation'
# animation.save(f'{fname}.gif')
# Image(filename=f"{fname}.gif")
from IPython.display import Image
from PIL import Image as PILImage

def _make_supervised_animation(net, loaders, device, max_iters, name,normalize):
    global cfg

    if os.path.exists(f'{name}_diff_inter.gif'):
        print(f"skip '{name}'")
        return

    test_iterations = list(range(1,max_iters))
    out_acc,all_outs,all_iterin, _input, _output = test_default2(net, loaders["test"], test_iterations, 
                                                                 cfg.problem.name, device, short=True)
    if cfg.problem.name == 'mazes':
        d = {f'{name}_predicted': ANIMATION_TYPES.PREDICTED,
                f'{name}_diff_inter': ANIMATION_TYPES.DIFF_INTER,
                f'{name}_diff_inter_last': ANIMATION_TYPES.DIFF_INTER_1_and_LAST,

        }
    else:
        d = {
            # f'{name}_predicted': ANIMATION_TYPES.PREDICTED_FLAT,
                f'{name}_diff_inter': ANIMATION_TYPES.DIFF_INTER,
                f'{name}_diff_inter_last': ANIMATION_TYPES.DIFF_INTER_1_and_LAST,

        }

    B=_output.shape[0]
    B = np.random.randint(B)


    input_image = _input[B].permute(1,2,0).cpu().numpy()
    
    PILImage.fromarray((input_image*255).astype(np.uint8)).save(f'{name}_input.png')

    for fname, type in d.items():
        animation=_make_animation((out_acc,all_outs,all_iterin, _input, _output),
                                  type,B=B,frames=max_iters-2,interval=5,normalize=normalize)

        animation.save(f'{fname}.gif', fps=20)
        print(f"saved animation -{fname}.gif")


def eval_supervised_animation(run,name,size=13,max_iters=201,file_name='model_best',normalize=True,test_batch_size=5):
    run_id = run.config['_content']['run_id']

    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id),test_size=size,file_name=file_name,test_batch_size=test_batch_size)

    _make_supervised_animation(net, loaders, device, max_iters, name, normalize)




### train checklist

from notebook_tools import get_wandb_plot_values

def eval_train_checklist(run):
    run.summary["train_check_acc"] = max(get_wandb_plot_values(run,f'Accuracy/acc')[1])==100
    run.summary["train_check_acc90"] = max(get_wandb_plot_values(run,f'Accuracy/acc')[1])>=90
    run.summary["train_check_acc99"] = max(get_wandb_plot_values(run,f'Accuracy/acc')[1])>=99
    run.summary["train_check_val_acc"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc')[1])==100
    run.summary["train_check_val_acc90"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc')[1])>=90
    run.summary["train_check_val_acc99"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc')[1])>=99
    try:
        run.summary["train_check_val_acc_OT"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc_overthinking')[1])==100
        run.summary["train_check_val_acc90_OT"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc_overthinking')[1])>=90
        run.summary["train_check_val_acc99_OT"] =max(get_wandb_plot_values(run,f'Accuracy/val_acc_overthinking')[1])>=99
    except:
        pass



import gymnasium as gym
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.wrappers import *


from gymnasium.envs.registration import register


def preprocess_obs(obs):
    obs = torch.from_numpy(obs.transpose(0,3,1,2)/10).float()
    # obs = torch.from_numpy(obs/10)

    obs = obs.to(device,memory_format=torch.channels_last)
    
    return obs



from enum import IntEnum
class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6

def convert_str_to_dataset_action(action:str):
    if action == 'forward':
        return 0
    elif action == 'rotate':
        return 1
    elif action == 'pickup':
        return 2
    elif action == 'toggle':
        return 3
    else:
        raise ValueError(f"Invalid action '{action}'")

class SimpleActionWrapper(ActionWrapper):
    """
    0 -> forward
    1 -> rotate right
    2 -> pickup
    3 -> toggle 

    """

    def action(self, action):
        if action == 0:
            return Actions.forward
        elif action == 1:
            return Actions.right
        elif action == 2:
            return Actions.pickup
        elif action == 3:
            return Actions.toggle
        else:
            raise ValueError(f"Invalid action '{action}'")



class SameStateTerminate(Wrapper):
    """
    Terminate the episode if the agent reaches the same state twice.
    """

    def __init__(self, env):
        """A wrapper that terminates the episode if the agent reaches the same state twice.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.visited_states = set()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.visited_states.clear()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # `isdoorClose_iskeyThere_rotacao_posAw_posAh`
        agent_pos = self.unwrapped.agent_pos
        agent_dir_vec = self.unwrapped.dir_vec

        is_door_present= False
        is_key_present = False

        door_pos = None
        key_pos = None
        goal_pos = None

        ## wall is always in the same place
        for o in self.unwrapped.grid.grid:
            if o is not None:
                if o.type =='door':
                    is_door_present = not o.is_open
                    door_pos = o.cur_pos
                elif o.type == 'key':
                    is_key_present = True
                    key_pos= o.cur_pos
                elif o.type == 'goal':
                    goal_pos=o.cur_pos
                elif o.type == 'agent':
                    assert agent_pos == o.cur_pos
        
        state = f"{is_door_present}_{is_key_present}_{agent_dir_vec[0]}_{agent_dir_vec[1]}_{agent_pos[0]}_{agent_pos[1]}"

        if state in self.visited_states:
            terminated = True
        else:
            self.visited_states.add(state)

        return obs, reward, terminated, truncated, info




def eval_env(run,run_id,size=13,max_iters=100,episodes=100,env='doorkey',summary_name='evals/env_{}',file_name='model_best',folder_path='../../../outputs/mazes_ablation',test_batch_size=25):
    start_time = time.time()
    # ## no idea why
    # if '_content' in run.config:
    #     run_config = run.config['_content']
    # else:
    #     run_config = run.config

    # run_id = run_config['run_id']


    net,loaders = load_model_and_dataloaders(run, get_model_path(run_id,folder_path),test_size=size,file_name=file_name,test_batch_size=test_batch_size)

    # out = supervised_test(net,loaders,max_iters=max_iters)
    
    assert env=='doorkey', 'not implemented env'

    ENV_NAME = "MiniGrid-Deep-Thinking-DoorKey-v0"
    register(
        id=ENV_NAME,
        entry_point="minigrid.envs:DoorKeyEnv",
        kwargs={"size": size},
    )


    env: MiniGridEnv = gym.vector.make(
        ENV_NAME,
        tile_size=1,
        num_envs=test_batch_size,
        wrappers=[FullyObsWrapper,ImgObsWrapper,SameStateTerminate,SimpleActionWrapper],
        # render_mode="human",
        # agent_pov=args.agent_view,
        # agent_view_size=args.agent_view_size,
        # screen_size=args.screen_size,
    )

    ## lets try this.... should be by default
    net.to(memory_format=torch.channels_last)
    net.eval()

    curr_reward = torch.zeros(test_batch_size)
    active_envs = np.ones(test_batch_size)
    episodic_rewards = []

    o,_ = env.reset()
    inputs = preprocess_obs(o)

    while len(episodic_rewards)<episodes:

        # get action
        with torch.no_grad():
            all_outputs = net(inputs, iters_to_do=max_iters)

        last_outputs = all_outputs[:, -1]
        action = get_predicted(inputs, last_outputs, 'doorkey').squeeze()
        
        assert len(action.shape)==1, f'predicted {action.shape}'

        # step env
        o, r, d, t,_ = env.step(action.cpu())

        inputs= preprocess_obs(o)

        curr_reward+=r

        save_r = np.where(((d+t)*active_envs)>0)[0]
        if len(save_r):
            print(f"finished {len(save_r)} episodes with returns",curr_reward[save_r].tolist())
            episodic_rewards+=list(curr_reward[save_r])
            curr_reward[save_r]=0
            episodes_left = episodes-len(episodic_rewards)

            if episodes_left < test_batch_size:
                disable_envs_idxs = save_r[:test_batch_size - episodes_left]
                active_envs[disable_envs_idxs]=0
        
            print(episodes_left," episodes left, ", active_envs.sum(), " active envs" )

    summary_name = summary_name.format(size)


    run.summary[summary_name+'_max'] = max(episodic_rewards)
    run.summary[summary_name+'_min'] = min(episodic_rewards)
    run.summary[summary_name+'_mean'] = np.mean(episodic_rewards)
    run.summary[summary_name+'_std'] = np.std(episodic_rewards)
    
    run.summary[summary_name+'_time'] = time.time()-start_time

    print(f"it took {time.time()-start_time} seconds")
    print(f"mean reward {np.mean(episodic_rewards)} std {np.std(episodic_rewards)}")

    # x_values, y_values = np.arange(len(out)), out
    # plot_name = plot_name.format(size)

    # # run.update()

    # idx = np.argmax(y_values)

    # run.summary[plot_name+'_max'] = y_values[idx]
    # run.summary[plot_name+'_max_iter'] = x_values[idx]

    # run.summary[plot_name+'_final'] = y_values[-1]


    # wandb_save_plot(run, plot_name,x_values, y_values)

