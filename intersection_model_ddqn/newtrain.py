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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=42,
        help="seed of the experiment")
    parser.add_argument("--exploration-fraction", type=float, default=0.1,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    
    parser.add_argument("--total_timesteps", type=int, default=1000)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--train_frequency", type=int, default=10)
    parser.add_argument("--target_network_frequency", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--start-e", type=float, default=0.3)
    parser.add_argument("--end-e", type=float, default=0.01)
    args = parser.parse_args()
    args.env_id = "intersection-v0"
    return args
args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_states = 968  
act_low = -1  
act_high = 1  
n_actions = 121 
def dis_to_con(discrete_action):
    ax = act_low + 0.2*(discrete_action%11)
    ay = act_low + 0.2*(discrete_action//11)
    return np.array([ax,ay])
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """comments:reduce the value of epsilon"""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

replay_buffer = ReplayBuffer(args.buffer_size)
model = Double_DQN(n_states, n_actions,
                 args.lr, args.gamma, device)
return_list=[]
time_list=[]
episodic_return = 0

if __name__ == '__main__':
    obs = env.reset()[0]
    env = RecordVideo(env, video_folder="intersection/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  
    start_t = time.time()
    for global_step in range(args.total_timesteps):

        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        state = obs.reshape([968])
        state = torch.Tensor(state[np.newaxis, :])
        action = model.take_action(state, epsilon)
        action1 = dis_to_con(action)
        next_obs, rewards, dones, infos, _ = env.step(action1)
        episodic_return += rewards
            
        if dones:
            end_t = time.time()
            episode_time = end_t - start_t
            time_list.append(episode_time)
            start_t = time.time()
            return_list.append(episodic_return)
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            episodic_return = 0
                
        replay_buffer.add(obs, action, rewards, next_obs, dones)
        if not dones: 
            obs = next_obs 
        else: 
            obs = env.reset()[0]
            
        if global_step > args.learning_starts and global_step % args.train_frequency == 0: 
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            transitions_dict = {
                    'states': s,
                    'actions': a,
                    'next_states': ns,
                    'rewards': r,
                    'dones': d,
                }
            model.update1(transitions_dict, args.batch_size)

            if global_step % args.target_network_frequency == 0:
                    model.update2()

    pickle.dump(model, open('model/ddqn_model.sav', 'wb'))
    env.close()
    plt.subplot(121)
    plt.plot(return_list)
    plt.title('return')
    plt.subplot(122)
    plt.plot(time_list)
    plt.title('time')
    plt.show()
        