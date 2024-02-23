import collections as col
import heapq
import os
import random

import matplotlib.pyplot as plt
import numpy as np


def gen_sample(h,w):
    """Generate empty grid, with player position and goal
    
    Input grid should be between 0 and 1
    """

    grid = np.zeros((3, h, w))

    # edges
    grid[:, 0, :] = 1
    grid[:, -1, :] = 1
    grid[:, :, 0] = 1
    grid[:, :, -1] = 1

    # player random position
    p_random_w, p_random_h = np.random.randint(1, w-1), np.random.randint(1, h-1)
    grid[0, p_random_h, p_random_w] = 1

    # goal random position
    g_random_w, g_random_h = np.random.randint(1, w-1), np.random.randint(1, h-1)
    
    #make sure different
    while (g_random_w, g_random_h) == (p_random_w, p_random_h):
        g_random_w, g_random_h = np.random.randint(1, w-1), np.random.randint(1, h-1)
    
    grid[1, g_random_h, g_random_w] = 1


    # compute vector from player to goal
    p_to_g = np.array([g_random_w - p_random_w, g_random_h - p_random_h])

    # compute optimal action (0,1,2,3) = (left, up, right, down)
    # where we first go up or down, and only then left or right
    
    ## this keeps the optimal policy without the problems of pixelation


    if p_to_g[1] > 0:
        optimal_action = 1 # up
    elif p_to_g[1] < 0:
        optimal_action = 3 # down
    else:
        if p_to_g[0] > 0:
            optimal_action = 2 # right
        else:
            optimal_action = 0 # left


    return grid, optimal_action



def gen_dataset(h,w, num_mazes):
    inputs = np.zeros((num_mazes, 3, h, w))
    targets = np.zeros((num_mazes))
    for i in range(num_mazes):
        inputs[i], targets[i] = gen_sample(h,w)
    return inputs, targets





if __name__ == "__main__":

    task_name = "goto"

    for size in [10,12,14,16,18,20,22,24,26,32,64,96,128,256,512,1024]:

        num_mazes = 50_000
        num_mazes_test= 10_000


        if size in [24,32,64,96,128]:

            num_mazes = 0
            num_mazes_test= 1_000

        elif size in [256,512,1024]:

            num_mazes = 0
            num_mazes_test= 100


        size_side=size
        size = (size,size)

        # num_mazes = 0
        # num_mazes_test= 1_000
        # l=80
        # size = (l,l)
        # for size in range(9, 18, 2):
        inputs_train, solutions_train = gen_dataset(*size, num_mazes)

        inputs_test, solutions_test = gen_dataset(*size, num_mazes_test)

        print(f"Mazes of size {size}, inputs.shape = {inputs_train.shape}, targets.shape = {solutions_train.shape}")

        if num_mazes > 0:
            data_name = f"data/{task_name}_data_train_{size_side}"
            if not os.path.isdir(f"{data_name}"):
                os.makedirs(f"{data_name}")
            ## change this to npz, much more space efficient.... Also in other maze stuff
            np.savez_compressed(os.path.join(data_name, "inputs.npz"), inputs_train)
            np.savez_compressed(os.path.join(data_name, "solutions.npz"), solutions_train)

        data_name = f"data/{task_name}_data_test_{size_side}"
        if not os.path.isdir(f"{data_name}"):
            os.makedirs(f"{data_name}")
        ## change this to npz, much more space efficient.... Also in other maze stuff
        np.savez_compressed(os.path.join(data_name, "inputs.npz"), inputs_test)
        np.savez_compressed(os.path.join(data_name, "solutions.npz"), solutions_test)

        if size_side<100:
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
        else:
            print("TOO big to compute repeats")


## For sizes in {9, 11, 13, 15, 17} we have 50,000 training examples and 10,000 testing examples. 
# For the larger sizes {19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 59}, we provide 1,000 testing mazes each.