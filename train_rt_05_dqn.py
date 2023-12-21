import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, ByteTensor
from collections import namedtuple
import random

import gym
import highway_env
from matplotlib import pyplot as plt
import numpy as np
import time

import gymnasium
import os
import math


task_code = "rt_05_dqn"
# Num of steps: 128->512
# Learning rate: 2e-4 -> 2e-4
# Total time steps: 500k -> 4m

task_folder = os.path.join("racetrack", task_code)
if not os.path.exists(task_folder):
    os.mkdir(task_folder)



def WriteServeral(folder, rd, ti, ch):
    name_list = ["reward.txt", "time.txt", "collision_hit.txt"]
    loss_list = [rd, ti, ch]
    for i in range(3):
        file = os.path.join(folder, name_list[i])
        with open(file, "a") as f:
            f.write(str(loss_list[i])+"\n")
            f.close()


Tensor = FloatTensor

EPSILON = 0  # epsilon used for epsilon greedy approach
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 40  # How frequently target netowrk updates
MEMORY_CAPACITY = 100
BATCH_SIZE = 80
LR = 0.01  # learning rate


class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        self.linear1 = nn.Linear(11, 11)
        self.linear2 = nn.Linear(11, 5)

    def forward(self, s):
        s = torch.FloatTensor(s)
        s = s.view(s.size(0), 11, 11)
        s = self.linear1(s)
        s = self.linear2(s)
        return s


class DQN(object):
    def __init__(self):
        self.net, self.target_net = DQNNet(), DQNNet()
        self.learn_step_counter = 0
        self.memory = []
        self.position = 0
        self.capacity = MEMORY_CAPACITY
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s, e):
        s = np.array(s[:1])
        x = np.expand_dims(s, axis=0)
        if np.random.uniform() < 1 - e:
            actions_value = self.net.forward(x)
            action = torch.max(actions_value, -1)[1].data.numpy()
            action = action.max()
        else:
            action = np.random.randint(0, 5)
        return action

    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # print("s:", s)
        # print("s shape:", np.array(s, dtype=object).shape)
        # print(torch.FloatTensor(s[:1]).shape)
        # exit(1099)
        # print("Is that OK?")
        self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(s[:1]), 0),
                                                torch.unsqueeze(torch.FloatTensor(s_[:1]), 0),
                                                torch.from_numpy(np.array([a])),
                                                torch.from_numpy(np.array([r], dtype='float32')))  #
        self.position = (self.position + 1) % self.capacity

    def get_sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        return sample

    def learn(self):
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.net.state_dict())
        self.learn_step_counter += 1

        transitions = self.get_sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # torch.cat(batch.state)
        # print("\nIs that OK?")
        b_s = Variable(torch.cat(batch.state))
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))

        # print("Forward:", self.net.forward(b_s).squeeze(1).shape)
        # print("Index:", b_a.unsqueeze(1).unsqueeze(1).shape)
        q_eval = self.net.forward(b_s).squeeze(1).gather(1, b_a.unsqueeze(1).unsqueeze(1).to(torch.int64))
        q_next = self.target_net.forward(b_s_).detach()  #
        # print("b_r:", b_r.shape)
        # print("q_next:", q_next.squeeze(1).max(1)[0].max(1)[0].shape)
        # Old one:
        # q_target = b_r + GAMMA * q_next.squeeze(1).max(1)[0].view(BATCH_SIZE, 1).t()
        # 我真是个天才
        q_target = b_r + GAMMA * q_next.squeeze(1).max(1)[0].max(1)[0].view(BATCH_SIZE, 1).t()
        loss = self.loss_func(q_eval, q_target.t())
        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step
        return loss


Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))



config = \
    {
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 15,
            "features": ["presence", "on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
            "grid_step": [5, 5],
            "absolute": False
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": False,
            "lateral": True,
        },

        "duration": 30,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
    }

env = gymnasium.make("racetrack-v0")
env.configure(config)

dqn = DQN()
count = 0

reward = []
avg_reward = 0
all_reward = []

time_ = []
all_time = []

collision_his = []
all_collision = []

NUM_EPOCH = 1000

for epoch in range(1, NUM_EPOCH+1):
    done = False
    start_time = time.time()
    s, info = env.reset()

    # Set default values in case of bugs
    avg_reward = -1
    avg_time = -1
    collision_rate = -1

    while not done:
        e = np.exp(-count / 300)  # 随机选择action的概率，随着训练次数增多逐渐降低
        a = dqn.choose_action(s, e)
        s_, r, done, ted, info = env.step(np.array([a, ]))
        env.render()

        dqn.push_memory(s, a, r, s_)

        if ((dqn.position != 0) & (dqn.position % 99 == 0)):
            loss_ = dqn.learn()
            count += 1
            print('trained times:', count)
            if (count % 50 == 0):
                avg_reward = np.mean(reward)
                avg_time = np.mean(time_)
                collision_rate = np.mean(collision_his)

                reward = []
                time_ = []
                collision_his = []


        s = s_
        reward.append(r)
        # print("Reward:", r)

    WriteServeral(task_folder, avg_reward, avg_time, collision_rate)

    end_time = time.time()
    episode_time = end_time - start_time
    print("Epoch", epoch, "ended with time", math.floor(episode_time), "seconds.")
    time_.append(episode_time)

    is_collision = 1 if info['crashed'] == True else 0
    collision_his.append(is_collision)

    if epoch%10==1:
        file_name = "model_rt_sd_" + str(epoch) + ".pt"
        save_path = os.path.join(task_folder, file_name)
        print("Saving model...")
        torch.save(DQNNet.state_dict(), save_path)
        print("Model saved at", save_path, ".")
