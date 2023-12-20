import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordVideo
import highway_env
from torch.distributions.normal import Normal
import numpy as np
import gymnasium

from train_rt_01_modified import Agent as trainAgent


device = 'cuda'

def make_highway_env(seed):
    def thunk():
        env = gymnasium.make("racetrack-v0", render_mode="rgb_array")
        env.configure({
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
        })
        env.reset()
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)
        #env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# env = gym.make("racetrack-v0", render_mode="rgb_array")
# , render_mode="rgb_array"
#

envs = gym.vector.SyncVectorEnv([make_highway_env(0)])

model_path = f"models\\model_rt_sd_10.pt"




"""env.configure({
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
        "longitudinal": False,
        "lateral": True,
    },

    "duration": 30,  # [s]
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
})
env.reset()"""



def act_inference(obs):
    #! Load Your own act_inferece
    # action = env.action_space.sample()
    # action, _ = model.predict(obs)
    actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
    action = actions
    return action


# print(env.action_space,env.observation_space)

if __name__ == '__main__':
    # Create the environment

    agent = trainAgent(envs).to(device)
    model_load = torch.load(model_path, map_location=device)
    agent.load_state_dict(model_load)
    agent.eval()
    print("Model loaded and evaluation mode on.")

    obs, info = envs.reset()

    # envs = RecordVideo(envs, video_folder="racetrack/videos", episode_trigger=lambda e: True)
    # envs.unwrapped.set_record_video_wrapper(envs)
    # envs.configure({"simulation_frequency": 15})  # Higher FPS for rendering


    cnt = 0

    for videos in range(1):
        done = truncated = False
        obs, info = envs.reset()
        while not (done or truncated):
            # Predict
            action = act_inference(obs)
            # Get reward
            obs, reward, done, truncated, info = envs.step(action.cpu().numpy())
            print("Reward:", reward)
            # Render
            # envs.render()
            cnt+=1
    envs.close()


# Things to do
# 1: Verify the effectiveness of this evaluation of vector env on a trained model
# 2: Try to write the socket of one ordinary env instead of vector env