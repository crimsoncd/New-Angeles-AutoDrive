import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import highway_env
from PPO_highway import PPO 
import os
import pygame
import torch
import numpy as np

pygame.init()

os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.display.set_mode((1200,800))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def make_env():
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "OccupancyGrid",
            "vehicles_count": 15,
            "features": ["presence", "on_road","x", "y", "vx", "vy", "cos_h", "sin_h"],
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

        "duration": 40,  # [s]
        "simulation_frequency": 15,  # [Hz]
        "policy_frequency": 1,  # [Hz]
    })
    env.reset()
    return env

def act_inference(obs,agent):
    #! Load Your own act_inference
    
    action = agent.take_action(obs)
    return action 

# print(env.action_space,env.observation_space)

if __name__ == '__main__':
    # Create the environment
    env = make_env()
    # print(env.action_space.shape[0],env.observation_space.shape[0])
    agent = PPO(n_states=env.observation_space.shape[0], 
            n_hiddens=16,  
            n_actions=env.action_space.shape[0],  
            actor_lr=1e-3,  
            critic_lr=1e-2,  
            lmbda = 0.95,  
            epochs = 10,  
            eps = 0.2, 
            gamma=0.9, 
            device = device
            )

    num_episodes = 10
    return_list = []

    for i in range(num_episodes):
    
        state = env.reset()[0]  # 环境重置
        done = False  # 任务完成的标记
        episode_return = 0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        for iter in range(10):
            action = agent.take_action(state)  # 动作选择
            next_state, reward, done, _, _  = env.step(action)  # 环境更新
            action = action.tolist()
            # print('state_update...')

            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state
            # 累计回合奖励
            episode_return += reward
            if done:
                time.sleep(0.5)
                break

        # 保存每个回合的return
        return_list.append(episode_return)
        # 模型训练
        agent.learn(transition_dict)

        # 打印回合信息
        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')
            
    obs, info = env.reset()
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = RecordVideo(env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15,
                   })  # Higher FPS for rendering

    for videos in range(1):
        done = truncated = False
        obs, info = env.reset()
        env.render()
        while not (done or truncated):
            # Predict
            action = act_inference(obs,agent)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()