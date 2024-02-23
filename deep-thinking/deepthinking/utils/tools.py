import logging
import random
from datetime import datetime

import torch
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau

import deepthinking.models as models
from .mazes_data import *
from .prefix_sums_data import prepare_prefix_loader, prepare_prefix_loader_MOD, prepare_prefix_loader_MOD_JOINT
from .chess_data import prepare_chess_loader
from .mnist_scale import prepare_mnist_loader
from .. import adjectives, names

from easy_to_hard_data import get_problem_dataset, GLOBAL_PROBLEM_NAMES

from .warmup import ExponentialWarmup, LinearWarmup


from typing import Tuple

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def generate_run_id():
    # random.randint is inclusive
    hashstr = f"{adjectives[random.randint(0, len(adjectives)-1)]}-{names[random.randint(0, len(names)-1)]}"
    return hashstr


def get_dataloaders(problem_args):
    loader = None
    if problem_args.name == "prefix_sums":
        loader =  prepare_prefix_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                     test_batch_size=problem_args.hyp.test_batch_size,
                                     train_data=problem_args.train_data,
                                     test_data=problem_args.test_data)
    

    elif problem_args.name == "mazes":
        loader =  prepare_maze_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)

    elif problem_args.name == "mazes_imitation":
        loader =  prepare_maze_imitation_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)
    elif problem_args.name[:5] == "fast_" and problem_args.name[5:] in GLOBAL_PROBLEM_NAMES:
        loader =  prepare_custom_loader_fast(get_problem_dataset(problem_args.name[5:]),train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)
    

    elif 'curriculum_' in problem_args.name or 'curriculumv' in problem_args.name:

        if 'curriculum_' in problem_args.name:
            prob_name = problem_args.name.split('curriculum_')[-1]
            curriculum_version=0
        else:
            prob_name = problem_args.name.split('curriculumv')[-1]
            curriculum_version=int(prob_name[0])
            prob_name = prob_name[2:]


        loader =  prepare_custom_loader_curriculum(get_problem_dataset(prob_name),train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data,
                                   curriculum_version=curriculum_version)



    elif problem_args.name in GLOBAL_PROBLEM_NAMES:
        loader =  prepare_custom_loader(get_problem_dataset(problem_args.name),train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)


    elif problem_args.name == "chess":
        loader =  prepare_chess_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                    test_batch_size=problem_args.hyp.test_batch_size,
                                    train_data=problem_args.train_data,
                                    test_data=problem_args.test_data)
    else:
        raise ValueError(f"Invalid problem spec. {problem_args.name}")
    
    if problem_args.hyp.single_batch:
        loader = make_single_batch_loader(loader,
                                    repetitions=problem_args.hyp.single_batch_repetitions)

    return loader

## this should be refactored #FIXME deprecated
def get_dataloaders_curriculum(problem_args,train_bits):
    if problem_args.name == "prefix_sums":
        return prepare_prefix_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                     test_batch_size=problem_args.hyp.test_batch_size,
                                     train_data=train_bits,
                                     test_data=problem_args.test_data)
    
    elif problem_args.name == "mazes":
        return prepare_maze_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=train_bits,
                                   test_data=problem_args.test_data)
    elif problem_args.name == "chess":
        return prepare_chess_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                    test_batch_size=problem_args.hyp.test_batch_size,
                                    train_data=train_bits,
                                    test_data=problem_args.test_data)
    else:
        raise ValueError(f"Invalid problem spec. {problem_args.name}")


def get_multiple_test_dataloaders(problem_args):

    if problem_args.name[:5] == "fast_":
        name = problem_args.name[5:]
    else:
        name = problem_args.name

    ##FIXME
    if name == "mazes":
        return prepare_test_maze_loader(
                                   example_number=problem_args.hyp.multiple_eval_nr_examples,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   test_data_sizes=problem_args.hyp.multiple_eval_test_sizes)

    elif name in GLOBAL_PROBLEM_NAMES:
        return prepare_test_custom_loader(get_problem_dataset(name),
                                   example_number=problem_args.hyp.multiple_eval_nr_examples,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   test_data_sizes=problem_args.hyp.multiple_eval_test_sizes)

    else:
        raise ValueError(f"Invalid problem spec. {problem_args.name}")


def get_model(model, width, max_iters, in_channels=3):
    model = model.lower() ## this is the model name in the model file, this will be called for the class
    net = getattr(models, model)(width=width, in_channels=in_channels, max_iters=max_iters)
    return net


def get_optimizer(optim_args, model_args, net, state_dict):
    optimizer_name = optim_args.optimizer.lower()
    epochs = optim_args.epochs
    lr = optim_args.lr
    lr_decay = optim_args.lr_decay
    lr_schedule = optim_args.lr_schedule
    lr_factor = optim_args.lr_factor
    warmup_period = optim_args.warmup_period
    weight_decay= optim_args.weight_decay

    early_patience = optim_args.early_patience
    scheduler_patience = optim_args.scheduler_patience

    if optim_args.lr_throttle:
        # Reducing the lr here for the recurrent layers helps with stability,
        # To date (July 21, 2021), we may only need this for maze models.
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = model_args.max_iters
        all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1
        all_params = [{"params": base_params}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = AdamW(all_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented.")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=0)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=warmup_period)

    if lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_schedule,
                                   gamma=lr_factor, last_epoch=-1)
    elif lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1, verbose=False)
    
    elif lr_decay.lower() == "reduce_on_plateau":
        ## change lr_factor to 0.5
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=scheduler_patience)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {lr_decay} not yet implemented.")
    
    earlystopping = EarlyStopping('min', patience=early_patience)

    return optimizer, warmup_scheduler, lr_scheduler, earlystopping


### FIXME dont use data parallel at least for analysys, check other motives
def load_model_from_checkpoint(problem, model_args, device, use_data_parallel=True,use_compile=True)->Tuple[torch.nn.Module,int, torch.optim.Optimizer]:
    model = model_args.model ## what a bad way to do this, is the model name in model file
    model_path = model_args.model_path
    width = model_args.width
    max_iters = model_args.max_iters
    epoch = 0
    optimizer = None

    in_channels = 3
    if problem == "chess":
        in_channels = 12
    elif 'mnist' in problem:
        in_channels = 1
    elif 'simple_pong' in problem:
        in_channels = 1
    elif 'snake_astar_v6' in problem or 'snake_new' in problem or "snake_bfs_v7" in problem:
        in_channels = 9

    net = get_model(model, width, in_channels=in_channels, max_iters=max_iters)
    net = net.to(device)
    if device == "cuda" and use_data_parallel:
        print("Using data parallel")
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]


    ### pytorch 2.0
    if use_compile and torch.__version__[0] == "2" and torch.cuda.get_device_capability()[0] >= 7:
        print("Using torch 2.0 compilation")
        net = torch.compile(net)
        # net = torch.compile(net,mode='reduce-overhead') ## it appears to be incorrect, doesnt learn

    return net, epoch, optimizer


def now():
    return datetime.now().strftime("%Y%m%d %H:%M:%S")



""" Learning utilities """
from functools import partial
from torch.optim import Optimizer

class EarlyStopping(object): # pylint: disable=R0902
    """
    Gives a criterion to stop training when a given metric is not
    improving anymore
    Args:
        mode (str): One of `min`, `max`. In `min` mode, training will
            be stopped when the quantity monitored has stopped
            decreasing; in `max` mode it will be stopped when the
            quantity monitored has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which training is stopped. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only stop learning after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.

    """

    def __init__(self, mode='min', patience=10, threshold=1e-4, threshold_mode='rel'):
        self.patience = patience
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        """ Updates early stopping state """
        current = metrics
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

    @property
    def stop(self):
        """ Should we stop learning? """
        return self.num_bad_epochs > self.patience


    def _cmp(self, mode, threshold_mode, threshold, a, best): # pylint: disable=R0913, R0201
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        """ Returns early stopping state """
        return {key: value for key, value in self.__dict__.items() if key != 'is_better'}

    def load_state_dict(self, state_dict):
        """ Loads early stopping state """
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold,
                             threshold_mode=self.threshold_mode)
