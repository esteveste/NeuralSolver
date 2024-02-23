import collections as col
import heapq
import os
import random

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
# import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from minigrid.wrappers import *

from gymnasium.envs.registration import register


ENV_NAME = "MiniGrid-Deep-Thinking-DoorKey-v0"

def get_goal_str_action(agent_pos,agent_dir_vec,goal_pos,only_touch=False):

    # compute vector from player to goal
    p_to_g = np.array([agent_pos[0] - goal_pos[0], agent_pos[1] - goal_pos[1]])

    # print('goal vector',p_to_g)

    forward_action = 'forward'
    if only_touch and sum(np.abs(p_to_g))==1:
        forward_action =  'touch'

    # we need to remove from the agent position

    if p_to_g[1] > 0:
        # optimal_action = 1
        if agent_dir_vec[1]==-1:
            return forward_action
        else:
            return 'rotate'
    elif p_to_g[1] < 0:
        # optimal_action = 3
        if agent_dir_vec[1]==1:
            return forward_action
        else:
            return 'rotate'
    else:
        if p_to_g[0] > 0:
            # optimal_action = 2
            if agent_dir_vec[0]==-1:
                return forward_action
            else:
                return 'rotate'
        else:
            # optimal_action = 0
            if agent_dir_vec[0]==1:
                return forward_action
            else:
                return 'rotate'



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



def get_next_dataset_action(self):
    '''env from make'''
    agent_pos = self.env.agent_pos
    agent_dir_vec = self.env.dir_vec

    is_door_present= False
    is_key_present = False

    door_pos = None
    key_pos = None
    goal_pos = None

    # # global vars...
    # do_2_forwards_after_door = True
    # forwards_done_after_door = 0

    
    ## wall is always in the same place

    for o in self.grid.grid:
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

    # print(agent_pos,key_pos,is_key_present,door_pos,is_door_present,goal_pos,)
    # print( self.env.dir_vec, self.env.agent_dir)
    
    if is_key_present:
        action = get_goal_str_action(agent_pos,agent_dir_vec,key_pos,only_touch=True)
        if action=='touch':
            action = 'pickup'
    elif is_door_present:
        action =  get_goal_str_action(agent_pos,agent_dir_vec,door_pos,only_touch=True)
        if action == 'touch':
            action = 'toggle'
    # elif do_2_forwards_after_door:
    #     forwards_done_after_door+=1
    #     action = 'forward'
    #     if forwards_done_after_door==2:
    #         do_2_forwards_after_door=False

    elif sum(np.array([agent_pos[0] - door_pos[0], agent_pos[1] - door_pos[1]]))<1:
        # print('2 steps')
        # do 2 forward steps after open door
        action = 'forward'
    else:
        action = get_goal_str_action(agent_pos,agent_dir_vec,goal_pos,only_touch=False)



    return convert_str_to_dataset_action(action)




def get_env_data(steps):


    env: MiniGridEnv = gym.make(
        ENV_NAME,
        tile_size=1,
        # render_mode="human",
        # agent_pov=args.agent_view,
        # agent_view_size=args.agent_view_size,
        # screen_size=args.screen_size,
    )

    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    
    env = SimpleActionWrapper(env)

    obs = []
    actions = []

    o,_ = env.reset()
    for step in range(steps):
        # env.render()
        action = get_next_dataset_action(env)

        obs.append(o)
        actions.append(action)

        o, r, d, _,_ = env.step(action)


        if d:
            o,_=env.reset()

    return obs, actions







import collections as col
import heapq
import os
import random

import matplotlib.pyplot as plt
import numpy as np

def gen_dataset(h,w,num_mazes):
    if num_mazes==0:
        return np.zeros((0,3,h,w)),np.zeros((0))
    
    obs,actions = get_env_data(num_mazes)

    obs = np.array(obs)
    obs = obs.transpose((0,3,1,2))
    obs = obs/10 #[0,1]
    actions  = np.array(actions)

    ## pre-shuffle the data
    index_shuf = list(range(len(actions)))
    random.shuffle(index_shuf)

    obs = obs[index_shuf]
    actions = actions[index_shuf]


    return obs, actions





if __name__ == "__main__":

    name = "doorkey"

    # we remove the 256 and 512 since they take a long time to generate
    # if you want to generate them, you can add them to the list
    test_size=[24,32,33,48,64,96,128,] #256,512] 

    for size in [6,8,10,12,15,17,20,*test_size]:

        num_mazes = 50_000
        num_mazes_test= 10_000

        if size in [256,512]:
            num_mazes = 0
            num_mazes_test= 100

        elif size in test_size:

            num_mazes = 0
            num_mazes_test= 1_000


        ## register new env
        register(
            id=ENV_NAME,
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": size},
        )

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
            data_name = f"data/{name}_data_train_{size}"
            if not os.path.isdir(f"{data_name}"):
                os.makedirs(f"{data_name}")
            ## change this to npz, much more space efficient.... Also in other maze stuff
            np.savez_compressed(os.path.join(data_name, "inputs.npz"), inputs_train)
            np.savez_compressed(os.path.join(data_name, "solutions.npz"), solutions_train)

        data_name = f"data/{name}_data_test_{size}"
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

