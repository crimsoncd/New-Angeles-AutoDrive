from xml.dom.pulldom import START_ELEMENT
import torch
from torch import nn
from torch.nn import functional as F
import time
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from tqdm import tqdm
import matplotlib.pyplot as plt
from newddqn import *
import argparse
import os
import random
import pickle
import newtrain

env = gym.make("intersection-v0", render_mode="rgb_array")
env.configure({
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence","on_road", "x", "y", "vx", "vy", "cos_h", "sin_h"],
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
        "longitudinal": True,
        "lateral": True,
    },

    "duration": 13,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
})
env.reset()

if __name__ == '__main__':
    loaded_model = pickle.load(open('model/ddqn_model.sav', 'rb'))
    
    obs, info = env.reset()
    env = RecordVideo(env, video_folder="intersection/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            state = obs.reshape([968])
            state = torch.Tensor(state[np.newaxis, :])
            action = loaded_model.take_action(state, 0)
            action1 = newtrain.dis_to_con(action)
            obs, reward, done, truncated, info = env.step(action1)
            env.render()
    env.close()