
import collections as col
import heapq
import os
import random

import matplotlib.pyplot as plt
import numpy as np


def gen_sample(h,w):
    grid=np.zeros((1, h,w))

    ## make pad
    pad_size = 1
    pad_w = np.random.randint(pad_size,w-pad_size)
    grid[:,-1,pad_w-pad_size:pad_w+pad_size+1]=1

    # ball position
    b_w,b_h = np.random.randint(0,w), np.random.randint(0,h-1)
    grid[:,b_h,b_w]=1

    if b_w<pad_w:
        action=0
    elif b_w==pad_w:
        action=1
    else:
        action=2



    assert pad_w-pad_size>=0 and pad_w+pad_size<w

    return grid, action



def gen_dataset(h,w, num_mazes):
    inputs = np.zeros((num_mazes, 1, h, w))
    targets = np.zeros((num_mazes))
    for i in range(num_mazes):
        inputs[i], targets[i] = gen_sample(h,w)
    return inputs, targets





if __name__ == "__main__":

    task_name="pong"

    for size in [6,7,8,9,10,12,15,20,25,32,33,43,59,64,128]: #[8,10,12,15,20,25,32]:

        num_mazes = 50_000
        num_mazes_test= 10_000

        if size in [13,23,32,33,43,59,64,128]:

            num_mazes = 0
            num_mazes_test= 1_000



        elif size in [256,512,1024]:

            num_mazes = 0
            num_mazes_test= 100

    # num_mazes = 50_000
    # num_mazes_test= 1_000
    # l=10


        # size = (size,size)

        # num_mazes = 0
        # num_mazes_test= 1_000
        # l=80
        # size = (l,l)
        # for size in range(9, 18, 2):
        inputs_train, solutions_train = gen_dataset(size,size, num_mazes)

        inputs_test, solutions_test = gen_dataset(size,size, num_mazes_test)

        print(f"Mazes of size {size}, inputs.shape = {inputs_train.shape}, targets.shape = {solutions_train.shape}")

        if num_mazes > 0:
            data_name = f"data/{task_name}_data_train_{size}"
            if not os.path.isdir(f"{data_name}"):
                os.makedirs(f"{data_name}")
            ## change this to npz, much more space efficient.... Also in other maze stuff
            np.savez_compressed(os.path.join(data_name, "inputs.npz"), inputs_train)
            np.savez_compressed(os.path.join(data_name, "solutions.npz"), solutions_train)

        data_name = f"data/{task_name}_data_test_{size}"
        if not os.path.isdir(f"{data_name}"):
            os.makedirs(f"{data_name}")
        ## change this to npz, much more space efficient.... Also in other maze stuff
        np.savez_compressed(os.path.join(data_name, "inputs.npz"), inputs_test)
        np.savez_compressed(os.path.join(data_name, "solutions.npz"), solutions_test)

        inputs = np.concatenate((inputs_train, inputs_test), axis=0)

        # Check for repeats
        t_dict = {}
        t_dict = col.defaultdict(lambda: 0)     # t_dict = {*:0}
        for t in inputs:
            t_dict[t.tobytes()] += 1            # t_dict[input] += 1

        repeats = 0
        for i in inputs:
            if t_dict[i.tobytes()] > 1:
                repeats += 1

        print(f"Maze size: {size} \n There are {repeats} mazes repeated in the dataset. ({repeats/(num_mazes+num_mazes_test)*100} %)")



## For sizes in {9, 11, 13, 15, 17} we have 50,000 training examples and 10,000 testing examples. 
# For the larger sizes {19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59}, we provide 1,000 testing mazes each.