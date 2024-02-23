
import torch
from torch.utils import data
import numpy as np
from easy_to_hard_data import MazeDataset, MazeImitationDataset

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115),
#     Unused import (W0611).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, W0611


def prepare_maze_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = MazeDataset("../../../data", train=True, size=train_data, download=True)
    testset = MazeDataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset, 
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


def prepare_maze_imitation_loader(train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = MazeImitationDataset("../../../data", train=True, size=train_data, download=True)
    testset = MazeImitationDataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


def prepare_custom_loader(class_dataset,train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = class_dataset("../../../data", train=True, size=train_data, download=True)
    testset = class_dataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


def _make_train_loader(sizes,class_dataset,train_batch_size, shuffle=True):

    trainloaders = []

    for size in sizes:

        train_data = class_dataset("../../../data", train=True, size=size, download=True)
        train_split = int(0.8 * len(train_data))
        trainset, valset = torch.utils.data.random_split(train_data,
                                                        [train_split,
                                                        int(len(train_data) - train_split)],
                                                        generator=torch.Generator().manual_seed(42))

        trainloader = data.DataLoader(trainset,
                                    num_workers=0,
                                    batch_size=train_batch_size,
                                    shuffle=shuffle,
                                    drop_last=True)
        
        trainloaders.append(trainloader)

    return trainloaders
    
 

def prepare_custom_loader_curriculum(class_dataset,train_batch_size, test_batch_size, train_data, test_data, shuffle=True, curriculum_version=0):
    train_data_size=train_data

    train_data = class_dataset("../../../data", train=True, size=train_data, download=True)
    testset = class_dataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    # trainloader = data.DataLoader(trainset,
    #                               num_workers=0,
    #                               batch_size=train_batch_size,
    #                               shuffle=shuffle,
    #                               drop_last=True)
    
    ## FIXME HARDCODED
    if hasattr(class_dataset,'prob_name') and 'pong' in class_dataset.prob_name:
        print('PONG CURRICULUM')
        # sizes = [6,7,8,9,10,12,15]
        sizes = [6,8,10,12,15,17,20]


    elif hasattr(class_dataset,'prob_name') and '1s_maze' in class_dataset.prob_name:
        print('Maze imitation CURRICULUM')
        sizes = [5,7,9,11,13,15,17,19]

    else:
        sizes = [6,8,10,12,15,17,20,22,25,27,30,35] # minigrid

    ## dont do more than train size
    sizes = [size for size in sizes if size<=train_data_size]

    trainloaders = _make_train_loader(sizes,class_dataset,train_batch_size,shuffle)

    if curriculum_version==0:
        print("JOINT trainloader")
        trainloader = JointLoader(trainloaders)
    elif curriculum_version==1:
        print(f"Curriculum version {curriculum_version}: CurriculumEpochSampleLoader with epoch_increase=1")
        trainloader = CurriculumEpochSampleLoader(trainloaders,epoch_increase=1)

    elif curriculum_version==2:
        print(f"Curriculum version {curriculum_version}: CurriculumEpochSampleLoader with epoch_increase=2")
        trainloader = CurriculumEpochSampleLoader(trainloaders,epoch_increase=2)

    elif curriculum_version==3:
        print(f"Curriculum version {curriculum_version}: SimpleCurriculumEpochLoader with epoch_increase=2")
        trainloader = SimpleCurriculumEpochLoader(trainloaders,epoch_increase=2)

    elif curriculum_version==4:
        print(f"Curriculum version {curriculum_version}: CurriculumEpochPercentSampleLoader with epoch_increase=2 sample_percent=0.2")
        trainloader = CurriculumEpochPercentSampleLoader(trainloaders,epoch_increase=2,sample_percent=0.2)

    elif curriculum_version==5:
        print(f"Curriculum version {curriculum_version}: CurriculumEpochPercentSampleLoader with epoch_increase=4 sample_percent=0.2")
        trainloader = CurriculumEpochPercentSampleLoader(trainloaders,epoch_increase=4,sample_percent=0.2)

    elif curriculum_version==6:
        print(f"Curriculum version {curriculum_version}: CurriculumEpochPercentSampleLoader with epoch_increase=4 sample_percent=0.2")
        trainloader = CurriculumEpochPercentSampleLoader(trainloaders,epoch_increase=8,sample_percent=0.2)


    else:
        raise NotImplementedError(f"curriculum_version {curriculum_version} not implemented")




    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders


def prepare_custom_loader_fast(class_dataset,train_batch_size, test_batch_size, train_data, test_data, shuffle=True):

    train_data = class_dataset("../../../data", train=True, size=train_data, download=True)
    testset = class_dataset("../../../data", train=False, size=test_data, download=True)

    train_split = int(0.8 * len(train_data))

    trainset, valset = torch.utils.data.random_split(train_data,
                                                     [train_split,
                                                      int(len(train_data) - train_split)],
                                                     generator=torch.Generator().manual_seed(42))

    
    trainloader = data.DataLoader(trainset,
                                  num_workers=0,
                                  batch_size=train_batch_size,
                                  shuffle=shuffle,
                                  drop_last=True)
    valloader = data.DataLoader(valset,
                                num_workers=0,
                                batch_size=test_batch_size,
                                shuffle=False,
                                drop_last=False)
    
    valloader = DoNBatchesLoader(valloader,4)

    testloader = data.DataLoader(testset,
                                 num_workers=0,
                                 batch_size=test_batch_size,
                                 shuffle=False,
                                 drop_last=False)

    loaders = {"train": trainloader, "test": testloader, "val": valloader}

    return loaders
 
 

        
class OneBatchLoaderRepetitions:
    def __init__(self,loader,repetitions):
        self.loader = loader
        self.batch = next(iter(loader))
        self.repetitions = repetitions
        self.counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.repetitions:
            self.counter += 1
            return self.batch
        else:
            self.counter = 0
            raise StopIteration

class DoNBatchesLoader:
    def __init__(self,loader,n):
        self.loader = loader
        self.n = n
        self.counter = 0
        self.iter = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.n:
            self.counter += 1
            return next(self.iter)
        else:
            self.counter = 0
            self.iter = iter(self.loader)
            raise StopIteration


class JointLoader:
    def __init__(self,loaders):
        self.loaders = loaders
        # self.batch = next(iter(loader))
        self.batches = [iter(loader) for loader in loaders]
        self.repetitions = len(loaders[0])
        self.counter = 0
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.repetitions:
            self.counter += 1

            idx = np.random.randint(0,len(self.loaders))

            try:
                batch = next(self.batches[idx])
            except StopIteration:
                self.batches[idx] = iter(self.loaders[idx])
                batch = next(self.batches[idx])


            return batch
        else:
            self.counter = 0
            raise StopIteration


class CurriculumEpochSampleLoader:
    def __init__(self,loaders,epoch_increase=1):
        self.loaders = loaders
        # self.batch = next(iter(loader))
        self.batches = [iter(loader) for loader in loaders]
        self.repetitions = len(loaders[0])
        self.counter = 0

        self.curriculum_initial = 0
        self.epoch_counter = 0
        self.epoch_increase = epoch_increase
        

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.repetitions:
            self.counter += 1

            # idx = np.random.randint(0,len(self.loaders))
            idx = np.random.randint(0,min(len(self.loaders),self.curriculum_initial+1))

            try:
                batch = next(self.batches[idx])
            except StopIteration:
                self.batches[idx] = iter(self.loaders[idx])
                batch = next(self.batches[idx])


            return batch
        else:
            # increase curriculum over epochs
            self.epoch_counter+=1
            if self.epoch_counter%self.epoch_increase==0:
                self.curriculum_initial+=1

            self.counter = 0
            raise StopIteration


class SimpleCurriculumEpochLoader:
    def __init__(self,loaders,epoch_increase=1):
        self.loaders = loaders
        # self.batch = next(iter(loader))
        self.batches = [iter(loader) for loader in loaders]
        self.repetitions = len(loaders[0])
        self.counter = 0

        self.curriculum_initial = 0
        self.epoch_counter = 0
        self.epoch_increase = epoch_increase
        

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.repetitions:
            self.counter += 1

            idx = min(len(self.loaders)-1,self.curriculum_initial)

            try:
                batch = next(self.batches[idx])
            except StopIteration:
                self.batches[idx] = iter(self.loaders[idx])
                batch = next(self.batches[idx])


            return batch
        else:
            # increase curriculum over epochs
            self.epoch_counter+=1
            if self.epoch_counter%self.epoch_increase==0:
                self.curriculum_initial+=1

            self.counter = 0
            raise StopIteration


class CurriculumEpochPercentSampleLoader:
    def __init__(self,loaders,epoch_increase=1,sample_percent=0.2):
        self.loaders = loaders
        # self.batch = next(iter(loader))
        self.batches = [iter(loader) for loader in loaders]
        self.repetitions = len(loaders[0])
        self.counter = 0

        self.curriculum_initial = 0
        self.epoch_counter = 0
        self.epoch_increase = epoch_increase

        self.sample_percent = sample_percent
        

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < self.repetitions:
            self.counter += 1

            # idx = np.random.randint(0,len(self.loaders))

            if np.random.rand()<self.sample_percent:
                idx = np.random.randint(0,min(len(self.loaders),self.curriculum_initial+1))
            else:
                idx = min(len(self.loaders)-1,self.curriculum_initial)

            try:
                batch = next(self.batches[idx])
            except StopIteration:
                self.batches[idx] = iter(self.loaders[idx])
                batch = next(self.batches[idx])


            return batch
        else:
            # increase curriculum over epochs
            self.epoch_counter+=1
            if self.epoch_counter%self.epoch_increase==0:
                self.curriculum_initial+=1

            self.counter = 0
            raise StopIteration


def make_single_batch_loader(loaders,repetitions):
    train,test,val = loaders["train"],loaders["test"],loaders["val"]

    print("WARNING SINGLE BATCH: train and validation are same batch")

    train = OneBatchLoaderRepetitions(train,repetitions)
    test = OneBatchLoaderRepetitions(test,1)
    val = OneBatchLoaderRepetitions(train,1)

    loaders = {"train": train, "test": test, "val": val}

    return loaders



def prepare_test_maze_loader(test_batch_size, example_number, test_data_sizes, shuffle=False):

    loaders = {}
    for test_size in test_data_sizes:
        testset_data = MazeDataset("../../../data", train=False, size=test_size, download=True)


        testset, _ = torch.utils.data.random_split(testset_data,
                                                        [example_number,
                                                        int(len(testset_data) - example_number)],
                                                        generator=torch.Generator().manual_seed(42))

        testloader = data.DataLoader(testset,
                                    num_workers=0,
                                    batch_size=test_batch_size,
                                    shuffle=False,
                                    drop_last=True) # be batch efficient
        
        loaders[test_size] = testloader

    return loaders

def prepare_test_custom_loader(class_dataset,test_batch_size, example_number, test_data_sizes, shuffle=False):
    loaders = {}
    for test_size in test_data_sizes:
        testset_data = class_dataset("../../../data", train=False, size=test_size, download=True)

        testset, _ = torch.utils.data.random_split(testset_data,
                                                        [example_number,
                                                        int(len(testset_data) - example_number)],
                                                        generator=torch.Generator().manual_seed(42))

        testloader = data.DataLoader(testset,
                                    num_workers=0,
                                    batch_size=test_batch_size,
                                    shuffle=False,
                                    drop_last=True) # be batch efficient
        
        loaders[test_size] = testloader

    return loaders