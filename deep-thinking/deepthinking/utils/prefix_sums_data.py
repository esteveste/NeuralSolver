
import torch
from torch.utils import data
from easy_to_hard_data import *

from math import ceil, floor

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_prefix_loader(train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, shuffle=True):

    dataset = PrefixSumDataset("../../../data", num_bits=train_data)
    testset = PrefixSumDataset("../../../data", num_bits=test_data)

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



def prepare_prefix_loader_MOD(problem_type,train_batch_size, test_batch_size, train_data, test_data,
                          train_split=0.8, shuffle=True):
    
    if problem_type == 'prefix_mod4':
        dataset_class = PrefixMod4Dataset

    elif problem_type == "prefix_mod3":
        dataset_class = PrefixMod3Dataset

    elif problem_type == "prefix_sums_last2":
        dataset_class = PrefixSumLast2Dataset

    elif problem_type == "prefix_sums_last4":
        dataset_class = PrefixSumLast4Dataset
    elif problem_type == "prefix_sums_last8":
        dataset_class = PrefixSumLast8Dataset

    elif problem_type == "prefix_sums_empty75":
        dataset_class = PrefixEmpty75Dataset

    elif problem_type == "prefix_sums_empty90":
        dataset_class = PrefixEmpty90Dataset
    
    else:
        raise ValueError(f"problem_type unknown {problem_type}")

    dataset = dataset_class("../../../data", num_bits=train_data)
    testset = dataset_class("../../../data", num_bits=test_data)

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

def prepare_prefix_loader_MOD_JOINT(problem_type,train_batch_size, test_batch_size, train_data_list:list, test_data,
                          train_split=0.8, shuffle=True):
    


    if problem_type == 'prefix_mod4_joint':
        dataset_class = PrefixMod4Dataset

    elif problem_type == "prefix_mod3_joint":
        dataset_class = PrefixMod3Dataset

    elif problem_type == "prefix_sums_last2_joint":
        dataset_class = PrefixSumLast2Dataset
    
    else:
        raise ValueError(f"problem_type unknown {problem_type}")

    testset = dataset_class("../../../data", num_bits=test_data)

    trainloader_list = []
    valloader_list = []
    for train_data in train_data_list:

        dataset = dataset_class("../../../data", num_bits=train_data)

        trainset, valset = torch.utils.data.random_split(dataset,
                                                        [int(train_split * len(dataset)),
                                                        int(len(dataset) - int(train_split * len(dataset)))],
                                                        generator=torch.Generator().manual_seed(42))

        trainloader = data.DataLoader(trainset, num_workers=0, batch_size=train_batch_size,
                                    shuffle=shuffle, drop_last=True)
        valloader = data.DataLoader(valset, num_workers=0, batch_size=test_batch_size,
                                    shuffle=False, drop_last=False)
        


        trainloader_list.append(trainloader)
        valloader_list.append(valloader)
    
    train_num_batches_dataset = floor(int(train_split * len(dataset)) /train_batch_size) #drop last
    test_num_batches_dataset = ceil(int(len(dataset) - int(train_split * len(dataset))) /test_batch_size)

    trainloader = MultipleDataLoader(trainloader_list, num_batches=train_num_batches_dataset)
    valloader = MultipleDataLoader(valloader_list, num_batches=test_num_batches_dataset)


    testloader = data.DataLoader(testset, num_workers=0, batch_size=test_batch_size,
                                 shuffle=False, drop_last=False)
    
    
    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders

### infinite loader
def infinite_loader(loader):
    while True:
        for data in loader:
            yield data

## finite loader
def finite_loader(infinite_loaders:list, num_batches):
    total_loaders = len(infinite_loaders)
    for i in enumerate(num_batches):
        yield next(infinite_loaders[i%total_loaders])


class MultipleDataLoader:
  
    def __init__(self, loaders,num_batches):
        assert all([isinstance(loader,data.DataLoader) for loader in loaders]), "all loaders must be DataLoader class"
        self.loaders = [infinite_loader(loader) for loader in loaders] 
        self.num_batches = num_batches
        self.total_loaders = len(loaders)
        self.current_loader = 0

        self.current_counter = 0
  

    def __iter__(self):
        self.current_counter = 0
        return self

    def __next__(self):
        if self.current_counter <= self.num_batches:
            self.current_counter += 1
            self.current_loader = (self.current_loader + 1) % self.total_loaders
            return next(self.loaders[self.current_loader])
        else:
            raise StopIteration