import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from collections import namedtuple
import gymnasium
import os
import time

# Constants and Transition definition
MEMORY_CAPACITY = 10000
LR = 0.01
BATCH_SIZE = 32
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 10

task_code = "rt_05_dqn_consult"

task_folder = os.path.join("racetrack", task_code)
if not os.path.exists(task_folder):
    os.mkdir(task_folder)

def WriteReward(folder, r):
    file = os.path.join(folder, "reward.txt")
    with open(file, "a") as f:
        f.write(str(r)+"\n")
        f.close

def RecordDeath(folder, e, r):
    file = os.path.join(folder, "death_loop_record.txt")
    with open(file, "a") as f:
        line = str(e) + ":" + str(r) + "\n"
        f.write(line)
        f.close()

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))

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

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = DQNNet()
        self.target_net = DQNNet()
        self.learn_step_counter = 0
        self.memory = []
        self.position = 0
        self.capacity = MEMORY_CAPACITY
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def forward(self, s):
        return self.net(s)

    def choose_action(self, s, e):
        s = np.array(s[:1])
        x = np.expand_dims(s, axis=0)
        if np.random.uniform() < 1 - e:
            actions_value = self.forward(x)
            action = torch.max(actions_value, -1)[1].data.numpy()
            action = action.max()
        else:
            action = np.random.randint(0, 5)
        return action

    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(s[:1]), 0),
                                                torch.unsqueeze(torch.FloatTensor(s_[:1]), 0),
                                                torch.from_numpy(np.array([a])),
                                                torch.from_numpy(np.array([r], dtype='float32')))
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

        b_s = Variable(torch.cat(batch.state))
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))

        q_eval = self.forward(b_s).squeeze(1).gather(1, b_a.unsqueeze(1).unsqueeze(1).to(torch.int64))
        q_next = self.target_net.forward(b_s_).detach()
        q_target = b_r + GAMMA * q_next.squeeze(1).max(1)[0].max(1)[0].view(BATCH_SIZE, 1).t()

        loss = self.loss_func(q_eval, q_target.t())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()

# Environment configuration
config = {
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
    "duration": 30,
    "simulation_frequency": 15,
    "policy_frequency": 1,
}

if __name__=="__main__":
# Create the environment
    env = gymnasium.make("racetrack-v0")
    env.configure(config)

    # Instantiate the DQN agent
    dqn_agent = DQN()

    # Training loop
    num_episodes = 1000

    for episode in range(1, num_episodes+1):

        time_start = time.time()
        print("Episode:", episode, end=", ")

        state, info = env.reset()
        total_reward = 0

        dead_loop_step = 1000

        while dead_loop_step>0:

            # In case of running into a dead loop:
            dead_loop_step -= 1

            # print("I'm not slacking off!")
            epsilon = max(0.1, 0.1 - 0.01 * episode)
            action = dqn_agent.choose_action(state, epsilon)
            next_state, reward, done, ted, _ = env.step([action, ])
            dqn_agent.push_memory(state, action, reward, next_state)

            if len(dqn_agent.memory) > BATCH_SIZE:
                loss = dqn_agent.learn()

            state = next_state
            total_reward += reward

            if done:
                break

        if dead_loop_step<=0:
            print("Perhaps ran death.")
            RecordDeath(task_folder, episode, total_reward)

        time_end = time.time()
        interval = math.floor(time_end - time_start)
        WriteReward(task_folder, total_reward)
        print("Completed using", interval, "seconds with reward", total_reward, ".")
        # print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        if episode%10==1:
            # Save the trained model
            model_name = "model_dpn_" + str(episode) + ".pth"
            model_file = os.path.join(task_folder, model_name)
            dqn_agent.save(model_file)
            print("Model saved.")

    # Later, when you want to use the trained model
    # Load the model
    # dqn_agent.load("trained_model.pth")

    # Now, you can use the dqn_agent to interact with the environment using the trained model
